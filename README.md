# Brainlet

A neural network from scratch in Julia without using any third-party libraries.

## Why Julia?

Julia is faster than Python while being similar enough in syntax. It has also has some niceties in the standard library that would require something like NumPy otherwise.

## Requirements

- Julia 1.10.12+

## Goals

1. Build basic training using matrices, weights, biases, an activation function (sigmoid initially), and a cost function.
2. Use finite differences to get training working and testable.
3. Implement backpropagation using partial derivatives.
4. Increase the difficulty of training problems: OR/AND gates, then a binary adder, then something more complex.

### More Complex Training Ideas

- Spiral classification: Classify points belonging to two interleaving spirals.
- Learn an image: Predict a small grayscale image's brightness from pixel coordinates.
- Handwritten digits: Recognize digits using MNIST.
- Find a transmitter: Predict a hidden transmitter's coordinates from noisy signal strengths at four fixed sensors.
- Imperfect Morse code: Recognize letters from pulse and gap durations, starting with clean timing and gradually adding noise.
- Bouncing ball: Given a ball's position and velocity in a rectangular arena, predict where it hits a wall after several bounces.
