# Contributing to NSGS

Thank you for your interest in contributing to the Neuro-Scheduling for Graph Segmentation (NSGS) project! This document provides guidelines and instructions for contributing.

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## How to Contribute

### Reporting Bugs

If you find a bug in the code or documentation, please submit an issue to our GitHub repository. Before submitting a new issue, please check if the bug has already been reported.

When submitting an issue, please include:
- A clear and descriptive title
- A detailed description of the bug
- Steps to reproduce the behavior
- Expected behavior
- Screenshots (if applicable)
- System information (OS, hardware, etc.)
- Any additional context

### Suggesting Enhancements

We welcome suggestions for enhancing NSGS. To suggest an enhancement:
1. File an issue describing your proposed enhancement
2. Explain why this enhancement would be useful
3. Outline the steps or implementation details if you have them

### Code Contributions

#### Setting Up Your Development Environment

1. Fork the repository
2. Clone your fork locally: `git clone https://github.com/your-username/nsgs.git`
3. Add upstream remote: `git remote add upstream https://github.com/amethystani/nsgs.git`
4. Create a branch for your work: `git checkout -b feature/your-feature-name`

#### Development Workflow

1. Make your changes, following our coding standards
2. Add tests for your changes when possible
3. Ensure all tests pass: `cd Backend && make test`
4. Update documentation as needed
5. Commit your changes with a clear commit message

#### Pull Request Process

1. Push your branch to your fork: `git push origin feature/your-feature-name`
2. Submit a pull request to the main repository
3. Ensure your PR description clearly describes the problem and solution
4. Link any relevant issues in the PR description
5. Wait for maintainers to review your PR
6. Address any feedback from the code review

## Coding Standards

### C++ Code Style

- Follow the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)
- Use meaningful variable and function names
- Comment non-obvious code sections
- Write descriptive header comments for functions

### JavaScript/React Native Code Style

- Follow the [Airbnb JavaScript Style Guide](https://github.com/airbnb/javascript)
- Use functional components with hooks for React components
- Write clean, self-documenting code
- Ensure mobile responsiveness

### Documentation

- Update documentation when changing code functionality
- Use Markdown for documentation files
- Keep API documentation up-to-date

## Testing Guidelines

- Write unit tests for new functionality
- Cover edge cases in your tests
- Ensure tests are deterministic
- For C++ code, use Google Test
- For JavaScript, use Jest

## License

By contributing to NSGS, you agree that your contributions will be licensed under the project's [MIT License](LICENSE).

## Questions?

If you have any questions about contributing, please reach out to the maintainers:
- Animesh Mishra - [am847@snu.edu.in](mailto:am847@snu.edu.in)

Thank you for helping improve NSGS! 