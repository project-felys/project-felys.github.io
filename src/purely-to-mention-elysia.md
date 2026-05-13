# Purely to Mention Elysia

I built a programming language just to tell people it's for Elysia.

**Links**: [Felys](https://www.felys.dev/en/compiler), [GitHub](https://github.com/project-felys/felys-lang)

## Technical Decisions

Programming languages are one of the most studied fields in computer science. The field is broad, ranging from low-level assembly to modern languages like Python. To make this language simple and easy to use while remaining dependency-free, the Felys programming language is:

- **Interpreted**: It would be too much work to emit code to multiple ISAs without using LLVM or existing software. Building a simple virtual machine that accepts custom bytecode is a pragmatic way.
- **Imperative**: The language is a product, so it needs to be easy to use. In contrast, functional languages are more for academic research and most people are not familiar with them, though this is probably something many computer science people learned in their first year. That said, Felys still embraces the functional paradigm for simplicity and clarity.
- **Dynamic Typing**: Designing a robust type system is complicated, so I kept it dynamic. This allows me to implement a neural network framework much more easily later.
- **Strong Typing**: Weak typing sometimes provides convenience, but it becomes problematic when programmers are not careful or the codebase is large. I personally don't like it.
- **Immutable**: This is a very functional concept, but the real reason I'm introducing it is to simplify memory management. Writing a garbage collector requires the use of `unsafe` Rust, and I want to avoid it. Immutability means that simple reference counting can provide acceptable performance.

Some of the problems can be easily solved by switching to another language like Go. Historically, the very first version was written in C and migrated to Rust. I stuck with Rust simply because it's a joy to see the compiler catch all the mistakes I made, and to learn how to code while fighting against the borrow checker and lifetime checker.

## Architecture

This entire project contains two major subsystems: the programming language (compiler and runtime) and an auto-differentiation system for neural network training.

### Programming Language

- **Front-end**: It parses source code strings into an abstract syntax tree. For fast iteration, a parser generator that bootstraps itself is introduced. The parser is driven by parsing expression grammar and deterministic finite automata algorithms.
- **Mid-end**: It converts the syntax tree to a linear control-flow graph in the form of static single assignment (SSA) intermediate representation (IR). There are several passes such as constant propagation, dead code elimination, and jump threading to optimize the code. Before emitting the bytecode, it also performs register allocation to minimize memory usage during runtime.
- **Back-end**: A virtual machine reads the bytecode and runs it. This is more similar to assembly, where data needs to be loaded and stored, and control flow is done via jumps and branches to offsets.

### Auto Differentiation

Recall that all variables are immutable; common designs like PyTorch do not work because they store and modify gradients inside the graph. This does not affect dynamic computation graph building, so native flow control is fully usable. The only difference is that gradients must be stored explicitly outside the graph. This is not great for an interface, but it does allow people to see the magic behind it.

## Apologies

I started this project in 2023. There is too much I could talk about, and I don't even know where to start. I will add more algorithm-related topics in the future. There are lots of coding bad practices, and I did not want to fix them until I worked in the industry. I am planning another round of reconstruction, but I'm unsure when I will start it.
