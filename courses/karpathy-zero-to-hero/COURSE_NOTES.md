# Course notes: karpathy-zero-to-hero

2 lectures. Timestamps link to the video.

> Built from SYNTHETIC sample transcripts (offline test data).

## Contents

1. The spelled-out intro to language modeling: building makemore
2. The spelled-out intro to neural networks and backpropagation: building micrograd

## The spelled-out intro to language modeling: building makemore

Video: https://www.youtube.com/watch?v=PaCmpygFfXo

**Summary.** In this lecture we build makemore, a character level language model, starting from the dumbest baseline: a bigram counting model. Always start with a simple baseline. A bigram model takes five minutes and gives you a number to beat.

**Key takeaways**

- In this lecture we build makemore, a character level language model, starting from the dumbest baseline: a bigram counting model. [00:00](https://www.youtube.com/watch?v=PaCmpygFfXo&t=0s)
- Always start with a simple baseline. A bigram model takes five minutes and gives you a number to beat. [00:12](https://www.youtube.com/watch?v=PaCmpygFfXo&t=12s)
- Look at your data with your own eyes; most bugs and most wins come from understanding the dataset. [00:38](https://www.youtube.com/watch?v=PaCmpygFfXo&t=38s)
- Sample from the model often. Qualitative samples tell you things that the loss number hides. [01:08](https://www.youtube.com/watch?v=PaCmpygFfXo&t=68s)
- Then we move to a neural network version and verify it reaches roughly the same loss as the counting baseline; that is a sanity check. [01:40](https://www.youtube.com/watch?v=PaCmpygFfXo&t=100s)
- Under time pressure I would rather have a small model that clearly works and demos well than a big ambitious one that is broken. [02:25](https://www.youtube.com/watch?v=PaCmpygFfXo&t=145s)
- Use a pretrained or simple model when you can; you do not need to train from scratch to show the idea. [02:36](https://www.youtube.com/watch?v=PaCmpygFfXo&t=156s)
- Do: start simple, verify every step, visualize, overfit a batch. Don't: add complexity before you have a baseline. [03:00](https://www.youtube.com/watch?v=PaCmpygFfXo&t=180s)

**Glossary**

- **model**: mentioned 7x
- **from**: mentioned 4x
- **baseline**: mentioned 4x
- **loss**: mentioned 4x
- **have**: mentioned 4x
- **simple**: mentioned 3x

## The spelled-out intro to neural networks and backpropagation: building micrograd

Video: https://www.youtube.com/watch?v=VMj-3S1tku0

**Summary.** Today we are going to build a tiny autograd engine from scratch so you really understand what is happening under the hood. My philosophy is that you should be able to build it yourself; if you can write it from scratch in a hundred lines you actually understand it.

**Key takeaways**

- Today we are going to build a tiny autograd engine from scratch so you really understand what is happening under the hood. [00:00](https://www.youtube.com/watch?v=VMj-3S1tku0&t=0s)
- Start with the simplest possible thing that works end to end, a single scalar value object, and only then add complexity. [00:23](https://www.youtube.com/watch?v=VMj-3S1tku0&t=23s)
- Check your gradients numerically. Nudge the input by a tiny h and compare against the analytic derivative; this catches a lot of bugs. [00:57](https://www.youtube.com/watch?v=VMj-3S1tku0&t=57s)
- Another classic bug: forgetting to zero the gradients before the backward pass. Your loss looks like it is training but it is wrong. [01:22](https://www.youtube.com/watch?v=VMj-3S1tku0&t=82s)
- Write a tiny test that compares your engine against PyTorch on the same expression; if both agree you can trust it. [01:50](https://www.youtube.com/watch?v=VMj-3S1tku0&t=110s)
- Get a working training loop first: forward pass, compute loss, backward, update. Then iterate on the model. [02:17](https://www.youtube.com/watch?v=VMj-3S1tku0&t=137s)
- Overfit a single tiny batch first. If you cannot drive the loss to near zero on four examples, something is broken. [02:52](https://www.youtube.com/watch?v=VMj-3S1tku0&t=172s)
- Things I warn against: copying large frameworks you do not understand, and adding features before the core loop works. [03:15](https://www.youtube.com/watch?v=VMj-3S1tku0&t=195s)

**Glossary**

- **tiny**: mentioned 4x
- **loss**: mentioned 4x
- **build**: mentioned 3x
- **understand**: mentioned 3x
- **what**: mentioned 3x
- **thing**: mentioned 3x
