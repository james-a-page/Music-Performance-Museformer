## Guided music performance-generation/score-performance translation model.

An encoder-decoder transformer model attempting to learn to generate musical performances making use of the Museformer attention scheme to improve accuracy.
We pre-train the decoder block of our model on the large [Giant-Midi dataset](https://github.com/bytedance/GiantMIDI-Piano) to improve the generated pieces quality while avoiding overfitting. And then make use of the [ASAP dataset](https://github.com/fosfrancesco/asap-dataset) to learn a translation between the midi-score or a piece and the midi recordings from different performances of the piece.

We make use of the Octuple tokenisation method, and use the [Museformer](https://github.com/microsoft/muzic/tree/main/museformer) attention scheme to reduce memory requirements and improve the quality of the pieces we generate.
