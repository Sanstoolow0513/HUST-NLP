# transformer+bert

```bash
# 训练
python train.py 
# 测试
python infer.py
```

## transformer
    训练正常，在infer时，执行如下
    ```
    python infer.py --model_path "saved_models_transformer\transformer_segmenter_best_f1.pth" --use_crf
    ```
    需要执行crf还有执行的给出具体的文件路径

## bert
    