---
layout: post
title: "Fucking Search Engines: How Do They Work? (By the Insane Clown Posse)"
modified:
categories: 
excerpt:
tags: [google, search engines, inverted index, query, term frequency, inverse document frequency, python]
comments: true
image:
  feature:
date: 2015-07-25T22:26:50-07:00
---
{% include _toc.html %}

It might just be me, but every time I use Quora, I end up seeing at least one question like [this one](http://www.quora.com/I-am-confident-that-I-am-going-to-build-a-search-engine-that-will-compete-with-Google-at-least-in-the-smallest-scale-possible-first-but-for-now-I-dont-know-any-programming-What-should-I-do): someone questioning how Google works, and how they can "beat" Google at search. Most of the questions aren't as brazen or misinformed as this one, but they all express a similar sentiment, and in doing so, they betray a significant misunderstanding of how search engines work. 

But while Google is incredibly complex, the basic concept of a search engine that matches and ranks results entirely by their relevance to a text query isn't particularly difficult, and can be understood by anyone with basic programming experience. I don't think it's possible to surpass Google in search at this point, but making a search engine is a very attainable goal, and is actually quite an enlightening exercise that I'd recommend trying. 

That's what I'll be describing in this blog post: how to make a search engine for local text files that can handle standard queries (at least one of the words in the query appears in the document) and phrase queries (an entire phrase appears in the text) and can rank using the basic tf-idf scheme. 

There are two main stages in developing this: building the index, and then using the index to answer queries. And then on top of this, we can add result ranking (tf-idf, PageRank, etc.), query/document classification, and maybe some Machine Learning to keep track of a user's past queries and selected results to improve the search engine's performance. 

So without further ado, lets begin!

Building the Index
------------------

So the first step in building a text search engine is assembling an inverted index. Let me explain what exactly that is. An inverted index is a data structure that maps tokens to the documents they appear in. In this context, we can consider a token to simply be a word, so an inverted index, at its most basic, is just something that takes in a word, and returns to us a list of the documents it appears in. 

First, however, we have to parse and tokenize (split into words) our corpus of documents. We'll do this as follows: for every document we want to add to our index, we'll remove all punctuation and split it on whitespace, and create a temporary hashtable that maps filenames to their list of tokens. We'll repeatedly transform this hashtable until we reach the final inverted index I described above (well, with one extra complication, but I'll explain this). Here's the code to do the initial text filtering: 

	def process_files(self):
		file_to_terms = {}
		for file in self.filenames:
			pattern = re.compile('[\W_]+')
			file_to_terms[file] = open(file, 'r').read().lower();
			file_to_terms[file] = pattern.sub(' ',file_to_terms[file])
			re.sub(r'[\W_]+','', file_to_terms[file])
			file_to_terms[file] = file_to_terms[file].split()
		return file_to_terms
		

Two things that I haven't done here but I'd recommend doing are removing all the stopwords (words like "the," "and," "a," etc. that don't add to a document's relevance) and stemming all the words (so running, runner, runs all become run) using an external library (this will slow it down considerably though). 

Now I know I said that our inverted index would map words to document names, but, we also want to support phrase queries: queries for not only words, but words in a specific sequence. For this, we'll need to know where in each document each word shows up, so we can check for order. I used each word's index in the tokenized list for a document as the position of the word in that document, so our eventual inverted index will look like this:
	
	{word: {documentID: [pos1, pos2, ...]}, ...}, ...}
	
instead of this:

	{word: [documentID, ...], ...}
	

So our first task, then, is to create a mapping of words to their positions for each document, and then combine these to create our complete inverted index. This is what that looks like:

	#input = [word1, word2, ...]
	#output = {word1: [pos1, pos2], word2: [pos2, pos434], ...}
	def index_one_file(termlist):
		fileIndex = {}
		for index, word in enumerate(termlist):
			if word in fileIndex.keys():
				fileIndex[word].append(index)
			else:
				fileIndex[word] = [index]
		return fileIndex
		
	
This code is fairly self explanatory: it takes in the whitespace-split list of terms in the document (in which the words are in their original ordering), and adds each to a hashtable, where the values are a list of positions of that word. We build this list up iteratively as we go through the list, until we've gone through all the words, leaving us with a table keyed by strings and mapping to a list of positions of those strings. 

Now, we need to combine these hashtables. I started this by creating an intermediary index of the format 

	{documentID: {word: [pos1, pos2, ...]}, ...}
	
which we will then transform to our final index. That's done here:

	#input = {filename: [word1, word2, ...], ...}
	#res = {filename: {word: [pos1, pos2, ...]}, ...}
	def make_indices(termlists):
		total = {}
		for filename in termlists.keys():
			total[filename] = index_one_file(termlists[filename])
		return total
		

This code is very simple: it just takes the results of the file_to_terms function, and creates a new hashtable keyed by filename and with values that are the result of the previous function, making a nested hashtable.

Then, we can actually construct our inverted index. Here's the code:

	#input = {filename: {word: [pos1, pos2, ...], ... }}
	#res = {word: {filename: [pos1, pos2]}, ...}, ...}
	def fullIndex(regdex):
		total_index = {}
		for filename in regdex.keys():
			for word in regdex[filename].keys():
				if word in total_index.keys():
					if filename in total_index[word].keys():
						total_index[word][filename].extend(regdex[filename][word][:])
					else:
						total_index[word][filename] = regdex[filename][word]
				else:
					total_index[word] = {filename: regdex[filename][word]}
		return total_index
		
So, lets go through this. First, we make an empty hashtable (python dictionary), and we use two nested for loops to iterate through every word in the input hash. Then, we first check if that word is present as a key in the output hashtable. If it isn't, then we add it, setting as its value another hashtable that maps the document (identified, in this case, by the variable filename) to the list of positions of that word. 

If it is a key, then we do another check: if the current document is in each word's hashtable (the one that maps filenames to word positions). If it is, we extend the current positions list with this list of positions (note that this case is left in only for completeness: it will never be hit because each word will only have one list of positions for every filename). If it is not, then we set the positions equal to the positions list for this filename. 

And now, we have our index. We can input a word, and be returned a list of the documents it appears in, and the list of positions it appears in within these documents. Now, we'll learn how to query this index.

Querying the Index
------------------

Okay, so there are two types of queries we want to handle: standard queries, where at least one of the words in the query appears in the document, and phrase queries, where all the words in the query appear in the document in the same order. 

Before we start, however, I'd recommend sanitizing the query like we sanitized the documents when we built the index by stemming all the words, making everything lowercase, and removing punctuation. I won't go into this, as it's trivial to do, but it should be done before executing the query.

[Note: in all the code examples below, every function will take in a variable called 'invertedIndex', which is generated according to the previous section]. 

We're going to implement standard queries first. A simple way to implement these is to split the query into words (tokens, as described above), get a list, for each word, which documents they appear in, and then union all of these lists. Here's how we'd query for one word:

	def one_word_query(word, invertedIndex):
		pattern = re.compile('[\W_]+')
		word = pattern.sub(' ',word)
		if word in self.invertedIndex.keys():
			return [filename for filename in self.invertedIndex[word].keys()]
		else:
			return []
			
This code is pretty basic. All we're doing here is sanitizing the input with a regular expression, and then returning a list of all the keys in the hashtable for that word in the index (which is just all the filenames that word appears in). 

Then a standard query is a very simple extension on top of this: we just aggregate lists and union them, as show here:

	def free_text_query(self, string):
		pattern = re.compile('[\W_]+')
		string = pattern.sub(' ',string)
		result = []
		for word in string.split():
			result += self.one_word_query(word)
		return list(set(result))
		
If you wanted to implement a query that ensure that every word in the query shows up in the final result list, then you should just use an intersection instead of a union to aggregate the results of the single word queries. That's fairly trivial to do, and I'll leave it as an exercise to the reader. 

The final type of query is a phrase query, which is a bit more involved, as we have to ensure the correct ordering of the words in the documents. Here's the code for this query (I'll explain after):

	def phrase_query(self, string):
		pattern = re.compile('[\W_]+')
		string = pattern.sub(' ',string)
		listOfLists, result = [],[]
		for word in string.split():
			listOfLists.append(self.one_word_query(word))
		setted = set(listOfLists[0]).intersection(*listOfLists)
		for filename in setted:
			temp = []
			for word in string.split():
				temp.append(self.invertedIndex[word][filename][:])
			for i in range(len(temp)):
				for ind in range(len(temp[i])):
					temp[i][ind] -= i
			if set(temp[0]).intersection(*temp):
				result.append(filename)
		return self.rankResults(result, string)
		
		
So again, we first start off by sanitizing the input query. Then, we run a single word query for every word in the input, and add each of these result lists to our total list. Then, we create a set called setted, which takes the intersection of the first list with all the other list (essentially taking the intersection of all of the lists), and leaves us with our intermediate result set: all the documents that contain all the words in the query. 

Then, we have to check for ordering. So, for every list in the intermediate results, we first make a list of lists of the positions of each word in the input query. Then (pay attention here!) we use two nested for loops to iterate through this list of lists. For every position in every list, we subtract a number _i_, which increases as we go through the list of lists. _i_ increments by 1 when we got through the list of lists. Now, remember that python lists preserve order, so this list of lists contains the position lists of every word in the original query in the order of the words in the original query. Then, if these words are in the proper order, and we subtract an integer, _i_, from every position in each position list, and _i_ increments by 1 as we go to each successive position list, then, if these phrases are in order, the intersection of all of these modified lists of lists must have a length of at least one. 

Let me give an example:

Lets say the phrase we're querying for is "the cake is a lie." Then, for a specific filename, assume these are the positions of each word:

	the : [23, 34, 15]
	cake: [19, 35, 12]
	is: [179, 36, 197]
	a: [221, 37, 912]
	lie: [188, 6, 38]
	

Then, our list of lists is:
	
	[[23, 34, 15], [19, 35, 12], [179, 36, 197], [221, 37, 912], [188, 6, 38]]
	

Now, we subtract _i_ from each element in each list, where _i_ is 0 for the first list, 1 for the second list, 2 for the third list, etc.

	[[23, 34, 15], [18, 34, 11], [177, 34, 195], [218, 34, 909], [184, 2, 34]]
	

Then, we take the intersection of all of these lists, which will leave us with one element, 34. This would then indicate that the phrase  "the cake is a lie" is in the file in order. And this is true: if we look at the original list, we see that the sequence 34, 35, 36, 37, 38 gives us our phrase. 

There's many more query options you could support (take a look at Google's [advanced search](http://www.google.com/advanced_search) for inspiration). You can try to implement some of these on your own. 

Our final step, then, is to implement a query parser that will allow us to combine different types of queries to get a single result set (like how you can type something like 'the cake "is a lie"' into Google, which combines standard queries (the entire thing), and phrase queries ("is a lie"). This is very simple to do, as well: just use delimiters (like quotes) to specify a certain type of query, run each smaller query separately, and then intersect all of these result sets to get the final list of documents. 

Ranking Results
---------------
The final step in building a search engine is creating a system to rank documents by their relevance to the query. This is the most challenging part, because it doesn't have a direct technical solution: it requires some creativity, and examination of your own use case. For this post, we'll be implementing tf-idf ranking (term frequency - inverse document frequency) to order our documents, which is one of the simpler approaches to ranking. There won't be any code for this part: once you understand the theory of tf-idf, implementing it is pretty simple, with most of the work done while we create our index. 

So what is term frequency, the first part of our ranking scheme? Well, it's exactly what it sounds like: the number of times each word shows up in a particular document. Term frequency, as a metric, doesn't account for order: it assumes that a document is just an order-ambivalent collection of tokens, and an accurate representation of it can be obtained by enumerating the number of times each token appears. This isn't an entirely accurate assumption, but it's widely used in the field of document classification. It's more formally known as the "bag of words" model.

One simple representation of the bag of words model is by using document vectors: that is, we decompose every document into a vector of length _N_, where _N_ is the total number of unique terms in that document, and each entry is the number of times that specific term appears in that document. For multiple documents, a more convenient way of defining _N_ is as the number of unique words in all documents in the search space. This allows us to represent every document as a vector, and lets all of these vectors have equal dimensionality.

But wait! There's currently a major flaw in our model. Consider these two documents: "let them eat cake" and "let them eat cake let them eat cake". These are exactly the same, except that the second is just the first one repeated, but their vector representations would be very different: [1, 1, 1, 1] compared to [2, 2, 2, 2]. To solve this, we convert every vector into a unit vector by dividing by its magnitude (calculated by taking the square root of the sum of the squares of each entry in a vector), "normalizing" it. This would turn our previous vectors into [1/2, 1/2, 1/2, 1/2] and [1/2, 1/2, 1/2, 1/2], making them exactly the same, which is what we intend. 

This, however, still isn't enough. Term-frequency's fatal flaw (it's _hamartia_, for any Greek tragedians reading) is that it views every term as equally important in representing documents. But this isn't true: the word "and" tells us a lot less about a document than do the words "shakespeare" or "chlamydia." But, the word "and" shows up significantly more in documents than "chlamydia" does (or at least it does in the things I read), which will create false similarity between documents (because they all have high occurrences of the word "and"). 

To mitigate this, we have to add something to our term-frequency ranking system: inverse document frequency. Lets define the document frequency as _D<sub>t</sub>_ of some term _t_ to be the number of times _t_ shows up in the entire search space (that is, all the documents we're indexing). Then, if we have a collection of _K_ documents, inverse document frequency (_I<sub>t</sub>_) of the same term _t_ is just _K_/_D<sub>t</sub>_: the total number of documents divided by the number of documents the term _t_ shows up in. 

There is one final caveat in this: it corrects too strongly. If we have 10 documents (_K_ = 10) and one word shows up once in the collection, and another word shows up ten times in the collection, then the second word is considered ten times more important than the first. The linear behavior of this function is too radical, and will artificially reduce similarity by over-correcting. To fix this, we just add a natural log function in front, which will flatten out our function, making its correction more gradual. Then, the final function looks like this: in a collection of _K_ documents, for some term _t_, _I<sub>t</sub>_ = ln(_K_/_D<sub>t</sub>_), where _D<sub>t</sub>_ is the document frequency of _t_ and ln is the natural log function. 

(Implementation note: As you may have noticed, neither of these quantities depend on the query, and both can (and should) be computed for every term and document beforehand).

Now, to combine term-frequency and inverse document frequency into one metric, we can just multiply their values for each term. That is, the weight of any given term in our collection (the value of the entry in our document vectors) is just _T<sub>t</sub>_ * _I<sub>t</sub>_ : the term frequency times the inverse document frequency. 

Then, if we have a collection containing  _K_ documents, and all of these documents have _N_ total unique terms. Then, our documents will be represented as vectors each of length _N_, where the value of each entry (which corresponds to a term) is the term frequency of that term in that document multiplied by the inverse document frequency of that term in the collection. Each vector will only have 0's for the terms that don't appear in this document (remember, our vector's represent all unique words in the collection. Inverse document frequency will never be 0, because it's a collection level metric.

We will then do the same for the query: we'll represent it as a vector in _N_ dimensional space, just like the documents, and calculate the tf-idf score for each term in the query. This will, obviously, be much more sparse than the documents themselves. 

Now, the only step left is to calculate the similarity score between the query and its result set, and rank the documents in the result set by this score. There are tons of different approaches to this, but what I'll use here is something called [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity), which essentially just takes the dot product of the query and each document vector in the result set and divides it by the product of the magnitudes of these two vectors, which returns the cosine of the angle between these vectors (read [this](http://stackoverflow.com/questions/6255835/cosine-similarity-and-tf-idf) StackOverflow question for clarification). This is a particularly useful metric because it doesn't take the magnitudes of the two vectors into account when computing similarity (as opposed to, say, Euclidean Distance), which is quite important when you have one very sparse vector (our query), and one significantly less sparse vector (our document). 

So, to rank results, this is what we'd do: 

1. First, we precompute the tf and idf scores for every term, and we build the _N_ length vector for each document, using the tf * idf of each term as the entries.
2. Then, we compute the query, and get a result set of matching documents (using previously described techniques).
3. After this, we compute the vector for the query, which is also of length _N_ and uses the tf * idf as each of its entries. 
4. Then, we calculate the similarity between the query and each document in the result set (using cosine similarity), and get a score for each document. 
5. We sort the documents by this score, and return them, in order. 

And boom, we're done!

Conclusion
----------
Making a search engine that scales to the size of Google is incredibly difficult. But, making a simple one for personal use (or even as a proof of concept) isn't that hard at all. Actually, the way search engines build indices, rank, and query documents is qute intuitive, and building one is an exercise worth doing. 

The bag of words model that we employed here shows up everywhere. Another great application for it is as a email spam filter (which I've also made, and will make a post on eventually), or as a document classifier, or an article recommender, or any number of things. It's a very cool concept, and it's worth thinking about how you'd use it to implement any of the above (or something cooler). 

Anyways, that's all from me for today. If you have any feedback, or questions, or you just want to comment on the reference in the title, leave a comment or drop me an email/facebook message/whatever other newfangled social network you kids use nowadays. 

[All the code used in the article (along with a ranking implementation)](https://github.com/logicx24/Text-Search-Engine) 

