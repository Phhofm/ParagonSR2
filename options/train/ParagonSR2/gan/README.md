# ParagonSR2 GAN / Experimental Configs

> [!WARNING]
> **Experimental Zone**: The configurations in this folder are **not final** and serve as examples or historical records of my experiments.

## Contents
- **`example_previous_versions.yml`**: An old config file showing how I trained previous iterations.
- **`loss.txt`**: Logs from previous experimental runs.

## Plans for the Future
I plan to add properly tuned GAN configurations here in the future, utilizing the **MUNet** discriminator. These will come in a later release.

## Training a GAN Model?
If you want to train a GAN model *now*, I recommend:
1.  Copy one of the `fidelity` configs.
2.  Add a Discriminator config section (use MUNet or UNetDiscriminatorSN).
3.  Add Perceptual Loss (LPIPS/DISTS) and Adversarial Loss (GAN Loss).
4.  Load your pre-trained Fidelity model using `pretrain_network_g`.
