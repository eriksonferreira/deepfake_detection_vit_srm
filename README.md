# Detecção de Deepfakes com Filtros de Alta Frequência em Arquitetura Híbrida CNN+ViT

Este repositório contém o código oficial referente à dissertação de mestrado intitulada "Detecção de Deepfakes com Filtros de Alta Frequência em uma Arquitetura Híbrida de Redes Neurais Convolucionais e Vision Transformers", realizada no Programa de Pós-Graduação em Computação Aplicada do Instituto Federal do Espírito Santo (IFES).

## Resumo

Nos últimos anos, o avanço das tecnologias de aprendizado profundo tornou possível a criação de deepfakes, conteúdos digitais manipulados capazes de imitar rostos, vozes e gestos com alta fidelidade. Embora tragam benefícios para entretenimento e comunicação, o uso indevido gera preocupações éticas e de segurança. Este trabalho investiga se a aplicação de filtros de alta frequência (SRM e Sobel) em uma arquitetura híbrida baseada em CNNs e Vision Transformers (CrossViT) pode aprimorar a detecção de deepfakes, especialmente em cenários cross-dataset. Modelos foram treinados com dados do Deepfake Detection Challenge (DFDC) e testados nas bases Celeb-DF v2 e FaceForensics++, demonstrando maior eficácia com o filtro SRM integrado na arquitetura CrossViT.

## Citação

Se você utilizar o código ou modelos deste repositório, por favor cite a dissertação:

```
@mastersthesis{ferreira2025deteccao,
  author = {Ferreira, Erikson Eler},
  title = {Detecção de Deepfakes com filtros de alta frequência em uma arquitetura híbrida de redes neurais convolucionais e vision transformers},
  year = {2025},
  school = {Instituto Federal do Espírito Santo},
  address = {Serra},
  note = {Dissertação (Mestrado em Computação Aplicada)},
  pages = {95 f.}
}
```

## Texto disponível no [repositório do IFES](https://repositorio.ifes.edu.br/handle/123456789/6192)

## Instalação

Para instalar as dependências do projeto:

```shell
pip install -r requirements.txt
```

Com conda:

```shell
conda create -n deepfake_detection python=3.8
conda activate deepfake_detection
conda install pytorch torchvision cudatoolkit -c pytorch -c nvidia
pip install -r requirements.txt
```

## Preparação dos Dados

Os dados utilizados são derivados dos seguintes datasets:
- [Deepfake Detection Challenge (DFDC)](https://ai.facebook.com/datasets/dfdc/)
- [Celeb-DF v2](https://github.com/yuezunli/celeb-deepfakeforensics)
- [FaceForensics++](https://github.com/ondyari/FaceForensics)

A estrutura de diretórios deve seguir o padrão:

```
/path/to/dataset/
  train/
    real/
      frame1.jpg
    fake/
      frame2.jpg
  val/
    real/
      frame3.jpg
    fake/
      frame4.jpg
```

## Modelos e Configurações

Utilizamos a arquitetura híbrida CrossViT, especificamente a versão CrossViT-18†, com duas ramificações (L-Branch e S-Branch). Para as entradas, foram exploradas diferentes configurações:

- Imagens originais em ambas as entradas (baseline)
- Filtro Sobel aplicado no pré-processamento ou diretamente nas ramificações (L-Branch ou S-Branch)
- Filtro SRM aplicado no pré-processamento ou diretamente nas ramificações

## Treinamento

Exemplo de comando para treinamento do modelo CrossViT com filtro SRM na ramificação L-Branch:

```shell
python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --model crossvit_18_dagger_224 --nb_classes 1 --epochs 200 --batch-size 96 --data-set DFDC --output_dir /home/eferreira/master/cross-vit/CrossViT/old_logs/17_sobel_0_5 --drop 0.5 --clip-grad 0.5 --pretrained --is_experiment
```


## Resultados

Melhores resultados obtidos:

| Configuração                      | Dataset          | Acurácia | F1-score | AUC   | Log Loss |
|-----------------------------------|------------------|----------|----------|-------|----------|
| SRM L-Branch                      | DFDC             | 0.752    | 0.773    | 0.847 | 0.595    |
| SRM L-Branch                      | Celeb-DF v2      | 0.718    | 0.788    | 0.755 | 0.655    |
| SRM Pré-Processamento             | FaceForensics++  | 0.676    | 0.701    | 0.755 | 0.681    |

Esses resultados demonstram que a integração do filtro SRM, especialmente na ramificação L-Branch da arquitetura CrossViT, proporciona a melhor performance para detecção de deepfakes em diferentes cenários.
