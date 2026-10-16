import torch

from policy.trainer import Trainer


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    trainer = Trainer(
        learning_rate=1e-4,
        batch_size=16,
        device=device,
        grad_decay=0.4,
        env_width=60.0,
        env_length=60.0,
        env_height=15.0,
    )

    trainer.train(epoch=50)


if __name__ == "__main__":
    main()
