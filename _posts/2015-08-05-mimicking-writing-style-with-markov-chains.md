---
layout: single
title: "Mimicking Writing Style With Markov Chains"
modified:
categories: 
excerpt:
tags: [python, markov chains, markov, natural language, monte carlo]
comments: true
author_profile: true
image:
  feature:
date: 2015-08-05T23:43:22-07:00
---

I'm not sure if you guys will remember this, but a year or so back, there was a Facebook application that went viral called "What Would I Say," which claimed to "learn" your writing style from all of your previous activity on facebook, and then make statuses that sound like you. It wasn't very good, if we remember properly: sometimes, it would put together a proper sentence, but mostly, it was just nonsensical, where small subsets of the output seemed to work, but the final sentence failed utterly.

Well, the way that worked was using something called a markov chain, which is formally defined as a memoryless, random process that uses a probability distribution to transition from different states. These have a wide array of uses beyond generating gibberish: they accurately model a number of thermodynamic processes, can make great spam email generators, are the basis of Google's famous PageRank algorithm, [modeling baseball](http://www.pankin.com/sabr37.pdf), and even serving as the basis of speech recognition software (not entirely true: these use something more complicated called a [hidden markov model](https://en.wikipedia.org/wiki/Hidden_Markov_model), which I won't get into here). 

In this post, we're going to co-opt What Would I Say's idea to create our own markov chain that will generate text in the style of any given corpus.

So, lets begin.

How Do They Work
----------------

As I described above, the central idea of a markov chain is that we want to decompose a process to a series of separate states, where the transitions between states are probabilistic and depend only on the current state (more astute readers might recognize that this sounds a lot like a Finite State Machine, and they'd be right: [markov chains can be represented by FSMs](http://stackoverflow.com/questions/4880286/is-a-markov-chain-the-same-as-a-finite-state-machine)). So, our first step is to define writing style in a manner that can be interpreted by a markov model. 

What exactly is writing style? Well, at its most basic, writing style is a combination of diction (word choice) and the ordering of these words; factors which are, to a certain degree, specific to each individual writer. A simple way to define our markov chain, then, is to have every unique word as a state, and have the possible transitions for that word to be all the words that appear after this word in our seed corpus. Our transition probabilities can just be represented as the number of times each transition in the transitions list appears in the corpus for any given state (or in simpler terms, the number of times a word in the transitions list appears in the corpus). 

This approach takes into account both order and word choice, but it isn't precise enough, and a markov model based on this approach (called a unigram-approach) will generate a lot of nonsense. Consider the word "the:" there are a tremendous amount of words that will follow it in any reasonably sized corpus, and it will be included in many transition lists, which will make our chain essentially choose random words whenever it encounters that word. The counter for this is to change our states from being single-words to a group of words that appear in sequence: bigrams, trigrams, all the way up to n-grams, where each state is a group of words that are in order in the corpus, and our transitions are a list of all the words that come after this group (we get our new state by removing the first word from the original state, and adding the transition word to the end of the original state). 

There are also problems with this, however: when you're using longer states, like trigrams or quadgrams, and you don't have a very long corpus, there will be very few transition states for a large number of states (think about it: as your states increase in length, they become more specific (unique), and thus show up fewer times in the corpus). This makes the text you generate very repetitive, and prevents it from accurately showcasing the author's style. 

I found, after some experimentation, that a good compromise was to use bigrams (sequences of two words) as our states. They seem to be specific enough that they don't generate nonsense, but aren't so specific that the generated text is repetitive. Another approach would be to use a hybrid model: use both bigrams and unigrams, choose a new state using each, and then choose from each chosen state. I encourage you to try this, but I won't include it in this tutorial (as it's trivial to implement once you have one of these models implemented). 

A very natural way to implement a markov chain is using a hash table (if you've been reading my blog, you'll notice that hash tables are used in implementing a lot of things), where we map every state to a list of words that come after it. Then, our data structure to represent our markov chain will look like this:

	{state : [word1, word2, ...], ...}

Or as a real example
	
	{(the, man) : [was, ate, murdered, ...], ...}

where our possible next states could be (man, was), (man, ate), (man, murdered). 

But this isn't entirely correct, because in this model, we treat every word as equally likely to show up next, and this is patently untrue, as the number of times they show up after this state is (most likely) different for each word. We actually have to store frequency information, where each frequency represents the number of times the word shows up after the current state, and is our probability (it isn't a "real" probability, of course, as it'll be greater than 1, but it behaves like one. If you want formal probabilities, you could always just divide the frequency by the total number of words that show up after this state (including duplicates), but these numbers will have the same relationship to each other as the frequencies, so there's no need. In most cases, frequencies can be considered equal to probabilities).

Then, our markov chain representation will look more like this:

	{state : {word1 : freq1, word2 : freq 2, ...}, ...}

and like this with the keywords filled in:

	{(the, man) : {was : 28,  ate : 9,  murdered : 2, ...}, ...}

Here, we're using another hashtable where we map the transition words to the frequency that they appear after this state. 

Then, it's very simple to use this data structure to generate text. Here's how it'll work:

1. First, we randomly choose a state to start from. This will be our seed.
2. Then, we look at the transition table for that state, and do a weighted random choice to choose a transition word, where the weights are the frequency.
3. Then, we add the current state to the output string, and build the new state. To build the new state, we simply combine the second word of the current state with the chosen transition word (in that order).
4. Repeat steps 2 and 3 with our new state. 

I'll give a contrived example. Consider this to be our index for our markov chain:

	{
		(the, man) : {was : 28,  ate : 9,  murdered : 2}, 
		(man, ate) : {pizza : 12, steak: 8}, 
		...
	}

This clearly not the full index for this corpus, but it'll be enough for the example. Lets say our first, randomly chosen state was (the, man). Then, we add this to our output string, which is now "the man". Then, lets say our model chooses "ate" as the transition word. Then our new state is just (man, ate): the last word of the first state and the new, chosen transition. We then add this to our output string, which is now "the man ate," and we repeat with the new transition word (either "steak" or "pizza"). 

Implementation
--------------

Now that we understand how a markov chain works, it's quite easy to implement one. Our first step will be to assemble the index I described above, which we can then use to generate text.

To begin, we'll have to read in a corpus and then tokenize it, or split it into basic words. For simplicity, I took an approach similar to my previous post on search engines: I considered a word to be a block of text surrounded on both sides by whitespace (where all punctuation has been stripped out). This isn't a very intelligent method of tokenizing; for example, we'd want to consider proper nouns, like the album name "Surfer Rosa" to be single tokens, but it will do for this purpose. 

Then, from our corpus, we will emit every group of three words, which will then be used to create our model. The idea is as follows: for a sentence like 

	"Doolittle was an amazing album," 

we'd emit 

	(Doolittle, was, an), (was, an, amazing), (an, amazing, album)

We'd do this for our entire corpus. Then, using a hashtable, we can very simply use these triples to assemble our model, where the first two words in the triple are the state (or the key to our hashtable), and the last word is added to the transitions table (with its count incremented). Here's what this function looks like (here, words is a list of tokens in the corpus):

	def triples(words):
		if len(words) < 3:
			return
		else:
			for w in range(len(self.words)-3):
				yield (self.words[w],self.words[w+1],self.words[w+2])

Now, using these triples, we can assemble our index. 

	def database(words):
			for w1, w2, w3 in triples(words):
				key = (w1, w2)
				index = {}
				if key in index:
					if w3 in index[key]:
						index[key][w3] += 1
					else:
						index[key][w3] = 1.0
				else:
					index[key] = {w3: 1.0}
			return index

Lets go through this. We first loop through all the triples and assign each element to a variable: w1, w2, and w3. We make a tuple out of the first two words, and use them as a key into our hashtable. Then, there are a few cases: if the key isn't already in the index (that is, we're adding a new state), we add it to the index, and as its value, we create another hashtable with one key, w3 (the third element in the triple), and with 1.0 as its value (as its only shown up once so far as a transition state). If it is in the index, there are two cases: that this transition state shows up in the key's hashtable, or it doesn't. If it does, we just increment that transition's count by 1, and if it doesn't, we add it to the table, with a count of 1. 

I'll illustrate this with our previous example. Our triples list is currently  
	
	[(Doolittle, was, an), (was, an, amazing), (an, amazing, album)]

Then, our model looks like this:

	{(Doolittle, was) : {an : 1}, (was, an) : {amazing : 1}, (an, amazing) : {album : 1}}

Now, our model is complete. Here's the method to generate text from it (I'll explain after):

	def text_gen(n, words, index):
		gen_words = []
		first1 = random.randint(0,len(self.words)-2)
		first, next = words[first1], words[first1+1]
		for w in range(n):
			choice = weighted_choice(index[(first, next)].items())
			gen_words.append(choice)
			first, next = next, choice
		gen_words = ' '.join(gen_words)
		return gen_words

Here, we first choose our seed state by taking two random sequential words from our corpus. Then, we enter our loop, where we want to generate text with _n_ words, and we choose our next state by taking a weighted random choice of the transitions of the current state (were weights are frequencies). We add the current state to the corpus, create our new state, and repeate. We then rebuild our string and return. 

Conclusion
----------

Markov Chains are really cool, very applicable, and really aren't all that hard to implement. Now that you've built one, there's plenty more you can do, either using this basic technique, or applying more sophisticated techniques to generate better text or pose as a real, spam filter-defying Nigerian Prince. 

[Here's all the code from this article](https://github.com/logicx24/RedditMarkovBot). I also wrote a Reddit Bot that, on command, will generate text in the style of a particular user, which is also included in this repository. I might make a short follow up explaining that, but I'd encourage you to look at it yourself: it's a pretty easy to follow use of praw, the python library that wraps the reddit API
