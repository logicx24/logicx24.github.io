---
layout: single
title: "Recommending Subreddits by Computing User Similarity: An Introduction to Machine Learning in Python"
modified:
categories: 
excerpt:
tags: [reddit, subreddits, machine learning, nearest neighbors, recommendation engine, recommendations, python]
author_profile: true
comments: true
image:
  feature:
date: 2015-07-17T00:37:27-07:00
---

Someone famous once said that if you click on the first link on every Wikipedia page, you'll end up at the Philosophy page. The idea is that Wikipedia articles are written to provide a general introduction to the topic in the first blurb you see, linking back to broader umbrella topics that encompass the current subject. The broadest umbrella of all is Philosophy (think about it: everything, in one perspective, can be considered a form of philosphy), and so, eventually, if you keep clicking in the early parts of the article, you'll reach the philosophy page.

This is generally true (I personally verified it, but that's a topic for a different post), but it got me thinking, and I think I have my own conjecture to offer: if you click on enough links on TechCrunch, you'll eventually end up on an article about a startup that does "machine learning." 

Machine Learning nowadays has become a buzzword: an almost meaningless sales term that marketers and recruiters drop when they want their ventures to sound impressive. It's everywhere. And because of this peculiar elevation of an otherwise obscure subject, the entire topic of machine learning seems to have gained this "aura" of impenetratability, where otherwise competent software engineers shrink from it.

One of the reasons I decided to write this post was to demystify this topic a bit. I by no means claim to be an expert on the subject: I'm just an undergrad who found a PDF of "Elements of Statistical Learning" and decided to implement a bit of what I read. But, I think my application is pretty interesting: it's a problem that a lot of people, at some point, have had to deal with, and it makes for a great introduction to an incredibly broad topic. 

The Concept
-----------
The problem I had, and wanted to solve, was finding new content on Reddit. For those who don't know (I'd assume most of the readers of a tech blog know and understand Reddit, but I'll explain it succinctly anyways), Reddit is a social news site that aggregates content and ranks it by user feedback (upvotes and downvotes). It's broken up into a number of different communities, called subreddits, and each of these focus on a specific topic or subject. For example, you have /r/programming, /r/science, /r/music, along with more creative ones, like /r/askreddit (where all posts must be general questions which the community can offer personal answer too).

The problem with Reddit is that while there's a tremendous amount of content on the site (almost 10000 active subreddits, each focusing on a different subject), it's very difficult to discover new content. There's no built in system to recommend you new stories or new subreddits, making a lot of the more specific, obscure communities incredibly difficult to find and participate in. 

Every Redditor has this issue at some point, but there's no real solution to it. I decided that I wanted to solve this issue, and maybe discover some new content for myself, so I set out to make a subreddit recommendation engine. 

My main idea (my thesis, if you will) was that I'd probably like subreddits that people similar to me subscribe to. This seems quite intuitive: we tend to weight recommendations given to us from other people by how similar we perceive that person's tastes are. Then, the question is, how do we determine similarity between two Reddit users?

Well, simple. A redditor is defined by the subreddits they subscribe to, so we should base similarity between two redditors on their subscriptions. 

Then, we can represent every redditor as a _N_ length vector, where _N_ is the total number of subreddits that exist on Reddit. So, every dimension (entry in the vector) represents a specifc subreddit, and contains either a 1 or a 0: a binary indicator specifying whether or not that Redditor subscribes to that specific subreddit or not. 

Now, we have all these vectors, each representing a user. How do we determine the similarity between any two? Well, they are all vectors, so a simple idea would be to simply find the euclidean distance between two vectors. This, then, represents how "different" two vectors (users) are, and so, the vectors that are close together represent two users that are very similar, and vectors that are far apart represent users that are less similar.  

A major issue with this strategy (formally known as k-Nearest-Neighbors) is the "curse of dimensionality:" that as the number of dimensions increase, the metric of euclidean distance becomes almost meaningless because the differences in distance between different pairs of samples become negligible. For simplicity, I stuck with this method for this post, but I outline ways to counter it at the end.

Gathering Data
--------------
Before we can bgin to recommend anything, we need to collect data on a large number of users and subreddits they subscribe too. I know above I mentioned decomposing users into vectors of length _N_, where _N_ is the total number of subreddits on the site, but this is impractical and will dilute the distance function by adding a lot of false similarity (because 95% of the vector will be 0's). So instead, I decided to just maintain the union of all subscribed subreddits of all users in the database, and use that as my _N_ value instead. 

Reddit doesn't actually allow us to directly access the subreddits a user subscribes to, so I had to get around this by looking at each user's last 1000 comments, and considering every subreddit they've commented on in those comments as a subscription. This imposes a bias against more read-based subreddits, and makes our recommendation engine tend to offer recommendations for subreddits that have active conversations over subreddits that mainly share content. Such a bias can't be avoided, but it's worth thinking about. 

What I did to get information on users was to use the Reddit API to grab the last 250 comments made on /r/all (which aggregates the top posts from all subreddits), and look at the past posts of the users that posted each of those comments. I stored the username and an array of subscribed subreddits in MongoDB. I also maintained the union of all subscribed subreddits of all users in the database, which I used to build my vectors. 

Here's the code to assemble the dataset, in Python. It heavily utilizes praw, a Python library that wraps the Reddit JSON API and makes it easier to interact with. (NOTE: I created a group of custom MongoDB functions that would make database interactions easier. They're all fairly self-explanatory. You can find them [here](https://github.com/logicx24/SubredditRecommendationEngine/blob/master/mongo.py). Use them for reference if needed).


	def getSubredditUsers(subreddit):
		"""
		Get the commentors in a subreddit. 
		"""
		client = MongoClient()
		reddit = praw.Reddit(user_agent="kNN Subreddit Recommendation Engine", handler=MultiprocessHandler())
		subreddit = reddit.get_subreddit(subreddit)
		comments = subreddit.get_comments(limit=250)
		
		currentUsers = allUsers(client)
		if currentUsers:
			found = [user['username'] for user in currentUsers]
		else:
			found = []
			
		users = []
		for comment in comments:
			if comment.author.name not in found:
				users.append({'user':comment.author.name})
				
		return tempBulkInsert(users, client)

	def getComments(username):
		"""
		Return the subreddits a user has commented in.
		"""
		try:
			unique_subs = []
			client = MongoClient()
			
			reddit = praw.Reddit(user_agent="kNN Subreddit Recommendation Engine", handler=MultiprocessHandler())
			user = reddit.get_redditor(username)
			
			subs = []
			for comment in user.get_comments(limit=250):
				if comment.subreddit.display_name not in subs:
					subs.append(comment.subreddit.display_name)
					
				insertSub(comment.subreddit.display_name, client)
				
			return insertUser(username, subs, client)
			
		except requests.exceptions.HTTPError as e:
			print e
			pass

	def getSubreddits():
		return ['all']

	def cron(user):
		client = MongoClient()
		if abs(datetime.datetime.utcnow() - user['updated']).days >= 1:
			return getComments(username)

	def main():
		subs = getSubreddits()
		pool.map(getSubredditUsers, subs)
		users = [user['user'] for user in tempUserList(MongoClient())]
		for user in users:
			getComments(user)

	if __name__ == "__main__":
		main()
 
Alright, so lets go through this. The getSubredditUsers function does exactly what it says: it gets the most recent comments on a subreddit (in this case, we're using /r/all to grab comments from all over reddit), and adds all the authors of these comments to the database, providing us with a list of redditors to base recommendations off of. 

The getComments function then takes in a user and gets all the subreddits they've commented on. It stores each of these in a list, and adds them to each user object (so we now have an array of subscribed subreddits for every user). It also adds every subreddit to the list of all unique subreddits we're maintaining in the database, if it isn't already present. 

Providing Recommendations
-------------------------
Now that we have a database full of users and the subreddits the subscribe to, we can actually start to provide recommendations to other users. We'll use the scheme I described earlier: create user vectors, find the _k_ closest neighbors (most similar users), and then recommend the subreddit that is shared the most between your neighbors that you haven't already subscribed too.  

First, we have to create user vectors. But that's simple:

	def createUserVector(username):
	
		client = MongoClient()
		
		user = queryUser(username, client)
		unique_subs = list(subreddits(client))
		vector = [0]*len(unique_subs)
		
		for i in range(len(unique_subs)):
			if unique_subs[i]['name'] in user['subreddits']:
				vector[i] = 1
				
		return vector

Here, we first pull the user object from the database, and extract its list of unique subreddits. We then make a list of 0's that is as long as the total number of unique subreddits subscribed to by users contained in the database. Then, we simply loop through this union list, and set a 1 whenever a subreddit in the union is part of a user's subscribed list. If we wanted to, instead of setting the vector entries to 1, we could use any number, which would weight matching subreddits significantly more. This is a parameter that takes a bit of tweaking, but for this article, I'll leave it as 1. 

Now we'll need to calculate the distance between any two users. This is also quite simple to do, given two usernames: we just create vectors out of each user, and then just apply euclidean distance to them. It looks like this:

	def vectorDistance(user1, user2):
	
		vector1 = createUserVector(user1)
		vector2 = createUserVector(user2)
		
		dist = 0
		for i in range(len(vector1)):
			dist += pow(vector1[i] - vector2[i], 2)
			
		return math.sqrt(dist)
	
After this, we need to find the user's _k_ closest neighbors. This, too, is simple: we simply find the distance between the target user (the one we're recommending subreddits to), and every other user, sort them, and return the _k_ users with the smallest distances. That's what this code does:

	def getNeighbors(username, k):
		client = MongoClient()
		
		distances = []
		for user in allUsers(client):
				
			dist = vectorDistance(username, user['username'])
			distances.append((user['username'], dist))
			
		distances.sort(key=operator.itemgetter(1))
		return distances[:k]

Then, we have to go through each of these similar users, and return the subreddit that they share the most but that the target doesn't subscribe to. That's done here:

	def getRecommendedSubreddit(username):
	
		client = MongoClient()
		neighbors = getNeighbors(username, 70)
		
		neighborUsernames = allUsersInArray([neighbor[0]  \ 
			for neighbor in neighbors], \ 
		client)
		
		targetSubscriptions = queryUser(username, client)['subreddits']
		
		allNeighborSubs = [sub for user in neighborUsers  \
			for sub in user['subreddits']]
			
		subredditFrequency = {
			subredditName : allNeighborSubs.count(subredditName) \
			for subredditName in set(allNeighborSubs) \
			if subredditName not in targetSubscriptions
		}
		
		return max(subredditFrequency, key=subredditFrequency.get)
		
This one is a bit complicated, so I'll go through it. First, we're getting the nearest neighbors to the target user (I chose 70 arbitrarily), and filter out the usernames (if you look at the getNeighbors function, you'll see that it returns a list of tuples in the form (username, distance\_from\_target) ). Then, we create an array of all of the target user's subscribed subreddits, and use a Python list comprehension to get a single list of all the subreddits that the neighbors subscribe to, repeats included. After this, we create a hashtable of format {subreddit_name : number of neighbors that subscribed to it}, where we include all subreddits that the target user didn't subscribe to. Then, to offer the recommendation, all we do is return the subreddit from the hashtable with the highest value.

 And now, we just need the final touch: a place for you to enter a target user:
 
	 def main(username):
		dataset.getComments(username)
		return getRecommendedSubreddit(username)

	if __name__ == "__main__":
		username = raw_input()
		print(main(username))

Congratulations! You just implemented k-Nearest-Neighbors!

Conclusion
----------
There's a number of issues with this method. First, as I mentioned before, it has a strong bias towards subreddits with active discussions over subreddits that are more read-focused. It also is diluted by the prevalence of the default subreddits (the subreddits that every redditor is subscribed to automatically on signup), which create a lot of false positives. It's also very difficult to accurately determine a user's subscriptions: reddit only allows you to look at the past 1000 comments (and this is terribly slow, so I cut it to 250), which isn't at all representative of a user's history on the site. 

Performance-wise, my current implementation of this is terrible. It's slow, it's single-threaded, it maintains large amounts of data in-memory, and it won't scale without a major rewrite. This was written more as a proof of concept than an actual, scalable recommendation system, so it's very far from being usable. 

But I think this project serves well as an introduction to the basic concepts of Machine Learning. We started with an intuitive idea: recommending subreddits by comparing the similarity between Redditors, and turned it into an actual, working recommendation system, and along the way, learned about acquiring and transforming data, turning mathematics into actual code, and implementing and training models. That's pretty cool. 

I'd encourage anyone reading this as an introduction to implement a similar system for themselves, or maybe, implement a better model. I think Naive Bayes would be a good next step. I also think there are a lot better distance metrics that you could try out and evaluate. 

Anyways, I hope this was clear and helpful. Next week, I'll write about making a simple search engine for text files. 

Also, here's all the code from the article on [GitHub](https://github.com/logicx24/SubredditRecommendationEngine).

 

