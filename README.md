**FILE STRUCTURE AND DOCUMENTATION**



This file briefly outlines the role of each Python script used for the SimCLR few-shot classification project.



| Filename | Purpose |

| 

| **train.py** | Main script containing the high-throughput SimCLR training loop and checkpointing logic. |

| **model.py** | Defines the core model architecture, including the ResNet-50 Encoder and the Projection Head. |

| **data\_loader.py** | Manages the on-the-fly, medical-safe augmentation pipeline and the PyTorch DataLoader for pre-training. |

| **eval\_imagenet.py** | Executes the Linear Probing evaluation for the ImageNet control benchmark. |

| **eval\_scratch.py** | Executes the Linear Probing evaluation for the Scratch control baseline. |

| **eval\_simclr.py** | Executes the final Linear Probing test using the SimCLR pre-trained weights, which are stored in **latest\_checkpoint.pth** |

| **run\_training.sh** | Slurm script used to submit the 72-hour 'train.py' job to the HPC cluster. |





To run the training loop, we just need to submit the **run\_training.sh** Slurm script to the Newton HPC. 

For evaluation, we can use **eval\_scratch.py, eval\_imagenet.py,**  or **eval\_simclr.py,** as required, by providing appropriate paths to the 20 image train dataset, and the test dataset.

