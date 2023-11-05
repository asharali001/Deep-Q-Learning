## Introduction

This file will guide you on how to run our application and provide some essential information about the setup. Our project revolves around game environments where our AI agent learns to play the game using reinforcement learning. We utilize Python, PyTorch, and Gym to make it all work.

## Getting Started

To run the application smoothly, you need to follow these steps:

### Prerequisites

1. **Visual Studio**: You'll need to have Visual Studio Code installed on your system. It is the primary development environment for our project.

2. **Python (Version 3.10)**: Make sure you have Python 3.10 installed on your machine. If it's not installed, you can download it from [Python's official website] (https://www.python.org/downloads/release/python-3100/).

3. **Python Extension for Visual Studio**: You'll also need to install the Python extension for Visual Studio. This extension enhances Python development capabilities within Visual Studio.

### Installing Dependencies

To run our application, you must install the necessary Python modules. Open a command prompt or terminal and execute the following commands:

```bash
pip install gym
pip install torch torchvision
pip install matplotlib ipython
pip install numpy
```

You might also need gym[all] or gym[atari] to run atari games

```bash
pip install gym[atari]
pip install gym[all]
```

We also need to install ALE. The ALE is natively supported by OpenAI Gym. Anytime you create an Atari environment it invokes the ALE in the background.

```bash
pip install ale-py
```

If you still have issues with the license after these try accepting the ROM licenses
```bash
pip install gym[accept-rom-license]
```

We also gym[others] for processing

```bash
pip install gym[other]
pip install opencv-python
```