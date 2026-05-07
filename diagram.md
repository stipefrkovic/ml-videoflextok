## AdaWorld
 ### Video Data Generation
We take an action with a random agent for 4 frames but only capture the last i.e. we skip the first 3. Now we have Frame 0 and Action 0. This is repeated with a new action, and so on, until the agent dies or the maximum number of steps is reached.
### Latent Action Generation
*Alessio verify*. We take two consecutive frames and feed them to the model. The model also appends two learnable action tokens (for each of the tokens) to the input. We take only the second action token from the output. This is fed to the VAE and reduced from 768 to 32 dimensions.
## VideoFlexTok
 ### Video Data Generation
We take a no-op action for 4 frames and record all. Then, we take an action (one out 5) from a random agent for 4 frames and record all. This makes 8 total frames, first 4 'without' an action and the second 4 with an action. This is repeated until the agent dies or the maximum number of steps is reached.
### Latent Action Generation
We take frames *t* to *t+15* and duplicate the first frame to have 17 frames. We feed this to the model and take tokens 1 and 2 from the output. Tokens 3 and 4 are causally dependent on the previous ones and we ignore them for now. Token 0 is not used right now but we should double check this. *Stipe verify*. These two tokens each have 256 dimensions with the intuition that they're ordered by importance (similar to PCA).

## Differences
- The videos are sampled differently!
- VideoFlexTok does not have action tokens directly; we currently take a no-action-frame and an action-frame token pair. Maybe the action-frame token alone is enough but we have to see.
- The losses between the two methods are different. AdaWorld learns latent actions by encoding and decoding frames based on them i.e. with a reconstruction loss directly influenced by actions. VideoFlexTok learns tokens that autoregressively deconstruct a video from the beginning, without the concept of actions.


```mermaid
flowchart LR
    subgraph Diagram1["AdaWorld"]
        direction LR
        F1["Frame 0"]
        F2["Frame 1"]
        subgraph Input["ViT Input"]
            direction TB
            P1(("Tokens 0"))
            P2(("Tokens 1"))
            A1(("a_0"))
            A2(("a_1"))
        end
        F1 --> P1
        F2 --> P2

        P1 --> ViT[["Spatio-Temporal ViT"]]
        P2 --> ViT
        A1 --> ViT
        A2 --> ViT

        subgraph ViTOut["ViT Output"]
            direction TB
            OutT(("Tokens 0"))
            OutT1(("Tokens 1"))
            OutA1(("a_0"))
            OutA2(("a_1"))
        end
        ViT --> OutT
        ViT --> OutT1
        ViT --> OutA1
        ViT --> OutA2

        OutA2 -.->|"768 dim"| BVAE[["β-VAE"]]

        subgraph Output["VAE Output: 1 latent × 32-dim"]
            Z(["Latent Action"])
        end
        BVAE -.-> Z
    end

    subgraph Diagram2["VideoFlexTok"]
        direction LR
        F0["Frame 0"] --> Z0
        G1["Frame 1<br/>Frame 2<br/>Frame 3<br/>Frame 4"] --> V1[["VAE"]] --> Z1
        G2["Frame 5<br/>Frame 6<br/>Frame 7<br/>Frame 8"] --> V2[["VAE"]] --> Z2
        G3["Frame 9<br/>Frame 10<br/>Frame 11<br/>Frame 12"] --> V3[["VAE"]] --> Z3
        G4["Frame 13<br/>Frame 14<br/>Frame 15<br/>Frame 16"] --> V4[["VAE"]] --> Z4

        subgraph VAEInput["Input"]
            direction TB
            Z0(("VAE frame 0"))
            Z1(("VAE frame 1"))
            Z2(("VAE frame 2"))
            Z3(("VAE frame 3"))
            Z4(("VAE frame 4"))
        end

        Z0 --> CE[["Causal Encoder"]]
        Z1 --> CE
        Z2 --> CE
        Z3 --> CE
        Z4 --> CE

        subgraph Tokens["Output: 5 tokens × 256-dim"]
            direction TB
            T0(["Token 0"])
            T1(["Token 1"])
            T2(["Token 2"])
            T3(["Token 3"])
            T4(["Token 4"])
        end
        CE --> T0
        CE --> T1
        CE --> T2
        CE --> T3
        CE --> T4
    end

    subgraph Legend
        direction TB
        LFrame["Frame"]
        LToken["Token"]
        LLearn["Learnable"]
        LLatent["Latent"]
        LModel["Model"]
    end

    classDef frame fill:#fff2cc,stroke:#b38600,color:#000
    classDef token fill:#d4e5ff,stroke:#1e3a8a,color:#000
    classDef learnable fill:#90EE90,stroke:#2d6a2d,color:#000
    classDef latent fill:#FFB6E1,stroke:#a33d7a,color:#000
    classDef module fill:#e0ccff,stroke:#4b0082,color:#000

    class F1,F2,F0,G1,G2,G3,G4,LFrame frame
    class P1,P2,OutT,OutT1,Z0,Z1,Z2,Z3,Z4,LToken token
    class A1,A2,OutA1,OutA2,LLearn learnable
    class Z,T0,T1,T2,T3,T4,LLatent latent
    class ViT,BVAE,V1,V2,V3,V4,CE,LModel module
```
