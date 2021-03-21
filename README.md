# DeepLUCIA
- Deep-LUCIA (Deep Learning-based Universal Chromatin Interaction Annotator) is a deep learning based model to predict chromatin loops from genomic and epigenomic profile without CTCF signal.

# FAQ
TBA

# Input data
- The input files of Deep-LUCIA are genomic and epigenomic profile of 5kb genomic loci.
- By default, Deep-LUCIA uses -log(p-value) of 12 epigenomic marks ( DNase, H2AFZ, H3K27ac, H3K27me3, H3K36me3, H3K4me1, H3K4me2, H3K4me3, H3K79me2, H3K9ac, H3K9me3, and H4K20me1 ). 
- For demonstration, the authors prepare the prebuilt matrix for 7 cell lines ( E017 : IMR90 , E116 : GM12878, E117 : HeLa , E119 : HMEC , E122 : HUVEC , E123 : K562 , E127 : NHEK ) and 1 human tissue ( E100 : Psoas Muscle )
    - To fetch prebuilt matrix, run this command 
``` bash prefetch.sh ```


# Training, evaluation, and prediction
- For training, run this command 
``` python deeplucia_train.py learn_config/all.val_set_06.test_set_07.n2p_001.num_pos_16.json ```
- For evaluation, run this command
``` python deeplucia_eval.py Model/trained_model.h5 learn_config/all.val_set_06.test_set_07.n2p_001.num_pos_16.json ```
- For prediction, run this command 
``` python deeplucia_fullscan.py Model/trained_model.h5 E100 chrX  ```


# Dependencies
- numpy=>1.18.2
- tensorflow>=2.1.0
- scikit-learn>=0.22.1
- PyFunctional>=1.4.1 ( https://github.com/EntilZha/PyFunctional )



