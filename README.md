# Simple Multi Agent Deep Reinforcement Learning (MADRL) Solution for Chess ♟️

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
<img src="https://www.pygame.org/docs/_static/pygame_tiny.png" height="35">

🏗️🚧 UNDER CONSTRUCTION 🚧🏗️

The goal of this project is to build an environment for the game of chess and apply the Proximal Policy Optimization (PPO) algorithm to solve it using different methods. The chess environment implemented in this project will not support the en passant and castling moves.

The project will explore two different methods for applying the PPO algorithm to the chess environment. The first method will involve training two different agents with separate neural networks. These two agents will compete against each other, with each agent learning from the other's moves. This approach is known as self-play and has been shown to be effective in training game-playing agents.

The second method will involve training a single agent with a single neural network. This agent will learn to play both sides of the chessboard, meaning it will learn to play as both white and black pieces. This approach is known as joint training and has the advantage of being more computationally efficient since it only requires one agent to be trained.

## Installation 💾

```bash
git clone git@github.com:mhyrzt/Simple-MADRL-Chess.git
cd Simple-MADRL-Chess
python3 -m pip install requirements.txt
```

## Contributing 🤝

Contributions to this project are welcome. If you have any ideas or suggestions, feel free to open an issue or submit a pull request.

## License 🔑

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Citation 📃

....

## TODO

- [ ] Complete Agents
- [ ] if checked and after move is still check it's not a vlid move
- [ ] one king can never attack another.
- [ ] Test Another scenario :)).