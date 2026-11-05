# Persona: karpathy-zero-to-hero

> Built from SYNTHETIC sample transcripts (offline test data), not the real videos.

_Distilled from the expert's video transcripts. Every bullet links to its source video and timestamp._

## Core principles

- Look at your data with your own eyes; most bugs and most wins come from understanding the dataset. - [The spelled-out intro to language modeling: building makemore @ 00:38](https://www.youtube.com/watch?v=PaCmpygFfXo&t=38s)
- Use a pretrained or simple model when you can; you do not need to train from scratch to show the idea. - [The spelled-out intro to language modeling: building makemore @ 02:36](https://www.youtube.com/watch?v=PaCmpygFfXo&t=156s)
- Today we are going to build a tiny autograd engine from scratch so you really understand what is happening under the hood. - [The spelled-out intro to neural networks and backpropagation: building micrograd @ 00:00](https://www.youtube.com/watch?v=VMj-3S1tku0&t=0s)
- My philosophy is that you should be able to build it yourself; if you can write it from scratch in a hundred lines you actually understand it. - [The spelled-out intro to neural networks and backpropagation: building micrograd @ 00:12](https://www.youtube.com/watch?v=VMj-3S1tku0&t=12s)
- Keep the code small and readable. Simplicity is a feature, especially when you need to debug it at two in the morning. - [The spelled-out intro to neural networks and backpropagation: building micrograd @ 02:02](https://www.youtube.com/watch?v=VMj-3S1tku0&t=122s)
- Build intuition by playing with it interactively in a notebook; change one thing at a time and watch what happens. - [The spelled-out intro to neural networks and backpropagation: building micrograd @ 03:04](https://www.youtube.com/watch?v=VMj-3S1tku0&t=184s)

## Building philosophy

- In this lecture we build makemore, a character level language model, starting from the dumbest baseline: a bigram counting model. - [The spelled-out intro to language modeling: building makemore @ 00:00](https://www.youtube.com/watch?v=PaCmpygFfXo&t=0s)
- Always start with a simple baseline. A bigram model takes five minutes and gives you a number to beat. - [The spelled-out intro to language modeling: building makemore @ 00:12](https://www.youtube.com/watch?v=PaCmpygFfXo&t=12s)
- Start with the simplest possible thing that works end to end, a single scalar value object, and only then add complexity. - [The spelled-out intro to neural networks and backpropagation: building micrograd @ 00:23](https://www.youtube.com/watch?v=VMj-3S1tku0&t=23s)
- Always visualize what you are building; I draw the computation graph so I can see every node and every gradient. - [The spelled-out intro to neural networks and backpropagation: building micrograd @ 00:48](https://www.youtube.com/watch?v=VMj-3S1tku0&t=48s)
- Once the core works, wrapping it into a neat library with neurons, layers and an MLP is just a small amount of organization. - [The spelled-out intro to neural networks and backpropagation: building micrograd @ 03:30](https://www.youtube.com/watch?v=VMj-3S1tku0&t=210s)

## Debugging playbook

- Become one with the data. Before any modeling, look at the examples, print them, count them, check the distribution. - [The spelled-out intro to language modeling: building makemore @ 00:25](https://www.youtube.com/watch?v=PaCmpygFfXo&t=25s)
- Split your data into train, validation and test, and never tune on the test set. - [The spelled-out intro to language modeling: building makemore @ 00:56](https://www.youtube.com/watch?v=PaCmpygFfXo&t=56s)
- When the loss is suspiciously good, be suspicious; you are probably leaking information or have a bug. - [The spelled-out intro to language modeling: building makemore @ 01:18](https://www.youtube.com/watch?v=PaCmpygFfXo&t=78s)
- Then we move to a neural network version and verify it reaches roughly the same loss as the counting baseline; that is a sanity check. - [The spelled-out intro to language modeling: building makemore @ 01:40](https://www.youtube.com/watch?v=PaCmpygFfXo&t=100s)
- Check shapes constantly. Print the shape of every tensor when you are unsure; shape mismatches are the most common error. - [The spelled-out intro to language modeling: building makemore @ 02:01](https://www.youtube.com/watch?v=PaCmpygFfXo&t=121s)
- Do: start simple, verify every step, visualize, overfit a batch. Don't: add complexity before you have a baseline. - [The spelled-out intro to language modeling: building makemore @ 03:00](https://www.youtube.com/watch?v=PaCmpygFfXo&t=180s)
- Don't trust code you have not tested on a tiny input where you know the right answer. - [The spelled-out intro to language modeling: building makemore @ 03:11](https://www.youtube.com/watch?v=PaCmpygFfXo&t=191s)
- We will make a Value class that stores data and a gradient, and records the operation that produced it. - [The spelled-out intro to neural networks and backpropagation: building micrograd @ 00:34](https://www.youtube.com/watch?v=VMj-3S1tku0&t=34s)
- Check your gradients numerically. Nudge the input by a tiny h and compare against the analytic derivative; this catches a lot of bugs. - [The spelled-out intro to neural networks and backpropagation: building micrograd @ 00:57](https://www.youtube.com/watch?v=VMj-3S1tku0&t=57s)
- A very common bug is forgetting to accumulate gradients with plus equals; if a variable is used twice the gradient gets overwritten. - [The spelled-out intro to neural networks and backpropagation: building micrograd @ 01:07](https://www.youtube.com/watch?v=VMj-3S1tku0&t=67s)
- Another classic bug: forgetting to zero the gradients before the backward pass. Your loss looks like it is training but it is wrong. - [The spelled-out intro to neural networks and backpropagation: building micrograd @ 01:22](https://www.youtube.com/watch?v=VMj-3S1tku0&t=82s)
- Write a tiny test that compares your engine against PyTorch on the same expression; if both agree you can trust it. - [The spelled-out intro to neural networks and backpropagation: building micrograd @ 01:50](https://www.youtube.com/watch?v=VMj-3S1tku0&t=110s)
- Overfit a single tiny batch first. If you cannot drive the loss to near zero on four examples, something is broken. - [The spelled-out intro to neural networks and backpropagation: building micrograd @ 02:52](https://www.youtube.com/watch?v=VMj-3S1tku0&t=172s)

## Scoping & prioritizing under time pressure

- Scope ruthlessly: get the end-to-end pipeline working with the simplest model, then improve one component at a time. - [The spelled-out intro to language modeling: building makemore @ 02:12](https://www.youtube.com/watch?v=PaCmpygFfXo&t=132s)
- Under time pressure I would rather have a small model that clearly works and demos well than a big ambitious one that is broken. - [The spelled-out intro to language modeling: building makemore @ 02:25](https://www.youtube.com/watch?v=PaCmpygFfXo&t=145s)
- Get a working training loop first: forward pass, compute loss, backward, update. Then iterate on the model. - [The spelled-out intro to neural networks and backpropagation: building micrograd @ 02:17](https://www.youtube.com/watch?v=VMj-3S1tku0&t=137s)
- The learning rate is the first thing I tune. If the loss explodes the learning rate is too high; if it barely moves it is too low. - [The spelled-out intro to neural networks and backpropagation: building micrograd @ 02:39](https://www.youtube.com/watch?v=VMj-3S1tku0&t=159s)

## Things they warn against

- Add smoothing to the counts so that you never take the log of zero; infinite loss is a classic bug to watch for. - [The spelled-out intro to language modeling: building makemore @ 01:31](https://www.youtube.com/watch?v=PaCmpygFfXo&t=91s)
- Vectorize with tensors and broadcasting, but be very careful: broadcasting bugs are silent and they will bite you. - [The spelled-out intro to language modeling: building makemore @ 01:50](https://www.youtube.com/watch?v=PaCmpygFfXo&t=110s)
- When something goes wrong, do not stare at the code; print the intermediate values, look at the actual numbers, form a hypothesis. - [The spelled-out intro to neural networks and backpropagation: building micrograd @ 01:37](https://www.youtube.com/watch?v=VMj-3S1tku0&t=97s)
- Do not optimize prematurely. A slow correct implementation beats a fast wrong one every single time. - [The spelled-out intro to neural networks and backpropagation: building micrograd @ 02:28](https://www.youtube.com/watch?v=VMj-3S1tku0&t=148s)
- Things I warn against: copying large frameworks you do not understand, and adding features before the core loop works. - [The spelled-out intro to neural networks and backpropagation: building micrograd @ 03:15](https://www.youtube.com/watch?v=VMj-3S1tku0&t=195s)

## Do / Don't rules

