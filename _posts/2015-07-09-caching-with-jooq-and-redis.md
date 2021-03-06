---
layout: single
title: "Caching in Java with JOOQ and Redis"
author_profile: true
modified:
categories: 
excerpt:
tags: [java, redis, jooq, sql, cache, database, queries, postgres, redshift]
comments: true
image:
  feature:
date: 2015-07-09T23:11:21-07:00
---

Database caching is, to the say the least, a difficult problem. Part of this is technical: making a finely-grained cache can take a lot of work, and requires a bit of creativity. But a cache is a performance optimization, and so, the main challenge is evaluating your use case, and creating a system that is optimal for your own needs. 
  
In this article, I'll be describing how to design and code a basic, table-level cache for SQL queries. Not everyone needs this. For a lot of applications, a simple, service-level cache could suffice, and would save a lot of work and future complication. This was my first project at my current internship (at Natero, [check us out](https://www.natero.com/)), and our usage patterns demanded a cache that would be keyed by queries and would store result objects that we could easily retrieve. These jobs mainly are used to calculate the many metrics we offer, as a customer success company, and our customers use these metrics to make major business decisions, so the cache also had to always be current. There couldn't be any latency in syncing writes. 

Libraries
---------

JOOQ is a library our entire codebase uses. It's not an ORM, like Hibernate or SQLAlchemy; rather, it provides you full access to the flexibility and customizability of SQL through an API similar to the ActiveRecord API, while saving you from the unmaintainable hassle of manually constructing SQL strings. 

I also used Jedis, a popular Java library, to interface with Redis. It's very simple to use, with almost no setup, works well with multithreaded environments (through a connection pool), and is integrated with Redis-cluster, which will make scaling easier. 


Intercepting Queries
--------------------

The first step to creating a query cache is intercepting queries after they've been executed but before they access the database. JOOQ offers us a convenient framework for doing exactly this: the MockConnection/MockDataProvider classes. These classes were designed originally for "mocking" a database, allowing for easy unit testing, but they work very well as interfaces between the application layer and the model (database or Redis). 

(Before I start this, however, I strongly recomend you to go to the [JOOQ documentation](http://www.jooq.org/doc/3.6/manual/getting-started/) and get a grasp of the basics of JOOQ. From here on out, I'll assume you understand, generally, how to write and execute queries in JOOQ).

A MockConnection is an object that is instantiated with a MockDataProvider and implements the default java.sql.Connection interface, meaning it can be used wherever a standard database connection is. The MockDataProvider, then, is an object that takes the place of the database in providing Result objects to queries. 

Lets take a look at this simple MockDataProvider:

	class ResultCache implements MockDataProvider {

		public Connection connection;
		public Jedis jedis;

		public MockResult[] execute(MockExecuteContext ctx) 
		throws SQLException {
		
			Result<?> result = null; 

			DSLContext db = DSL.using(connection);
			Query ctxQuery = db.query(ctx.sql(), ctx.bindings());
			String inlinedSQL = ctxQuery.getSQL(ParamType.INLINED);

			if (inlinedSQL.toLowerCase().matches("^(with\\b.*?\\bselect|select|explain)\\b.*?")) {

				if (!jedis.exists(inlinedSQL)) {
					result = db.fetch(ctx.sql(), ctx.bindings());
					jedis.set(inlinedSQL, serialize(result));
				} else {	
					result = deSerialize(jedis.get(inlinedSQL));
				}
				return new MockResult[] { 
					new MockResult(result.size(), result)
				};
			} else {
				db.execute(ctx.sql(), ctx.bindings());
				return new MockResult[] {
					new MockResult(0, result)
				};
			}
		}
	}

There's a lot here, so lets go through it slowly. Obviously, this is a vastly simplified version of my actual code: it doesn't handle many special cases, or batch queries, and has other issues. 

What we're creating here is a class that implements the standard MockDataProvider interface and has two instance variables: connection (database connection) and jedis (Redis connection). The real meat of the code is the execute() method, which is called internally by JOOQ on query execution.

Inside this method, we first create a DSLContext, and use it to obtain a sql string with the bind variables inlined. This will be the key into our cache. Then we have to decide whether the query is a read (select) or a write (insert/update/etc). If it's a write, we execute it and return a null result (invalidation will be described later).

If it's a read, we then have two cases: either the query is in the cache, and so the result can be returned easily, or the query isn't, and the result has to first be fetched from the database, and then added to the cache (the serialize() and deSerialize() functions are basic implementations of object serialization in Java).

Then, we return the result, wrapped in a MockResult array, at which point it's sent back to the original query execution object. 

This is how you'd instantiate a DSLContext that will hit your MockDataProvider:
	
	DSLContext db = DSL.using(
		new MockConnection(new ResultCache(connection, jedis))
	);

That DSLContext can be used as normal to build queries, and all of them will be routed to the ResultCache object. We've accomplished the first part of caching! Now we just need work on the harder part: invalidations.

Table Extraction
----------------
Now, invalidations are tricky. There's a major decision we have to make here: what level do we need to invalidate at? I chose the table level (that is, whenever an insert is performed, I invalidate all queries that either select from or are joined with that table), but if your application has something like a consistent pattern of inserting into a table and then selecting from that table, a table-based invalidation cache will end up being worthless. I won't go into all the possibilities for invalidations here, but it's something worth thinking about. 
  
First off, to invalidate by table, we'll need to figure out how to extract table names from every query. We could do this through string parsing, which is what I set up first, but that'll become tricky with indefinitely nested select queries and complicated join expressions (both of which I had to deal with; if those aren't used in your application, than this is a simple route you can choose). 

Instead, I used a slightly obscure JOOQ utility called a VisitListener, which allows you to access a query while it's being rendered from JOOQ's functions to actual SQL. I subclassed JOOQ's DefaultVisitListener and overrode the visitStart function, which is called right before JOOQ renders a table name, field, or nested query (in JOOQ terms, called a QueryPart). 

Here's the method I used to pull out the table name from write (inserts only, in this demo):

	public void visitStart(VisitContext ctx) {
	
		boolean isInsideInto = false;
		Clause[] clauseList = ctx.clauses();

		for (int i = 0; i < clauseList.length; i++) {
			if (clauseList[i] == Clause.INSERT_INSERT_INTO) {
				isInsideInto = true;
			} 
		}
		if (isInsideInto) {
			if (ctx.queryPart() instanceof Table) {
				String tableName = sqlString(ctx);
				invalidate(tableName);
			}
		}

In this code, the [Clause](http://www.jooq.org/javadoc/3.6.x/org/jooq/Clause.html) object is an enum describing the part of the query we are currently in, and clauseList tells us the tree of clauses that lead to our current position in the query (read the [docs](http://www.jooq.org/javadoc/3.6.x/org/jooq/Clause.html); basically, JOOQ decomposes every query into a tree-like structure, where clause objects branch in different directions until they reach queryParts, like tables or field names or nested queries). What that for loop does, then, is tell us what part of the query we're in (which parts of the tree have we traversed to get here). If we've traversed an INSERT\_INSERT_INTO, and the current QueryPart is a Table, then this must be the table being inserted into. 

Now that we've determined what tables our write query touches, we can invalidate all queries by involving that table using the invalidate method. I'll describe invalidation shortly, but for now, consider it magic. 

We also will need to extract tables from read queries so we know what tables they involve. I overrode the visitEnd method for that. Here's the code:

	public void visitEnd(VisitContext ctx) {

		boolean isSelect = false;
		String sql = sqlString(ctx);
		Clause[] clauseList = ctx.clauses();

		for (int i = 0; i < clauseList.length; i++) {
			if ((clauseList[i] == Clause.SELECT_FROM || clauseList[i] == Clause.TABLE_JOIN)) {
				isSelect = true;
			}
		}
		if (isSelect) {
			ArrayList<String> threadList; 
			threadList = (ArrayList<String>) Global.threadlocal.get();
			if (ctx.queryPart() instanceof Table) {
					threadList.add(sql);
			}
			Global.threadlocal.set(threadList);
		}
	}

So lets go through this. This is similar to above: we're going through the clauseList and checking if we're inside a particular clause. If we are, and the current QueryPart is a table, then it must be a table involved in a select query (either in the select from part, or as a joined table). 

The threadlocal code is a utility I had to use to allow table names that were extracted in this method to be passed to the ResultCache.java file so that they could be processed and used. There isn't a direct way for these two objects to communicate, as they are instantiated entirely separately, as you will soon see. So, the threadlocal syntax allows us to pass a variable (in this case, a list) between two objects, while ensuring that the thread that placed the object must be the same as the thread that accesses it after. 

Invalidations
-------------
The framework I used for invalidations was strikingly simple. Basically, you use the above code to extract tables, and pass those tables to ResultCache.java. Then, we create [lists in Redis](http://redis.io/topics/data-types), where the key is the table name, and the values in that list are the query strings that involve that table. One query can be part of many lists. 

We build these lists in ResultCache, where, every time a query is executed, we add it to each list associated with the tables we pull from threadlocal. 

For example, if we had this very simple query:

	SELECT * FROM languages

Then we'd add it to a Redis list that had a key "languages" and had as values all queries that involved the table "languages."

This, then, makes invalidations effortless. When the method invalidate(tableName) is called, we simply access the list of all queries that involve tableName, delete them from Redis, and remove them from that list. The cache, then, no longer holds any result objects for these queries, and so, they are effectively invalidated.

Here's the code for the invalidation method. It's very simple, and if you don't understand anything there, look at the [Jedis docs](http://tool.oschina.net/uploads/apidocs/jedis-2.1.0/redis/clients/jedis/Jedis.html).

	private void invalidate(String table) {
		long len = jedis.llen(table);
		List<String> tableKeys = jedis.lrange(table, (long) 0, len);
		for (String key : tableKeys) {
			jedis.del(key);
			jedis.lrem(table, 1, key);
		}
	}

And here's the code for creating the lists in Jedis (this goes inside the read portion of the if-statement in the execute method in ResultCache.java):

	if (Global.threadlocal != null) {
		for (String st : (ArrayList<String>) Global.threadlocal.get()) {
			jedis.lpush(st, inlinedSQL);
		}
	}

Conclusion
----------
This cache is by no means complete. There are a number of special cases that you'll have to handle that are specific to your application that'll make this messy, and there's several other optimizations you could implement to make this faster. But, for anyone wanting to write a query cache in Java, this is a great place to start. 



