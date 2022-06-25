# `ptls.trx_encoder`
All classes from `ptls.nn.trx_encoder` also available in `ptls.nn`

`ptls.nn.trx_encoder` helps to make a representation for single transactions.

## `ptls.nn.PaddedBatch`

Input data is a raw feature formats. You can transform your transaction to correct format with `ptls.data` module.
Common description or sequential data and used data formats are here [data_preparation.md](../data_preparation.md)
Input data are covered in `ptls.nn.PaddedBatch` class.

We can create `PaddedBatch` object manually for demo and test purposes.

```python
x = PaddedBatch(
    payload={
        'mcc_code': torch.randint(1, 10, (3, 8)),
        'currency': torch.randint(1, 4, (3, 8)),
        'amount': torch.randn(3, 8) * 4 + 5,
    },
    length=torch.Tensor([2, 8, 5]).long()
)
```

Here `x` contains three features. Two are categorical and one is numerical:

- `mcc_code` is categorical with `dictionary_size=10`
- `currency` is categorical with `dictionary_size=4`
- `amount` is numerical with `mean=5` and `std=4`

`x` contains 5 sequences with `maximum_length=12`. Real lengths of each sequence are `[2, 8, 5]`.

We can access `x` content via `PaddedBatch` properties `x.payload` and `x.seq_lens`.

Real data have sequences are padded with zeros. We can imitate it with `x.seq_len_mask`. 
It returns tensor with 1 if a position inside corresponded seq_len and 0 if position outside.
Let's check out example
```python
>>> x.seq_len_mask
Out: 
tensor([[1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 0, 0, 0]])
```
There are 2, 8 and 5 valid tokens in lines.

More way of `seq_len_mask` usage are in `PaddedBatch` docstring.

We can recreate our `x` with modified content:
```python
x = PaddedBatch({k: v * x.seq_len_mask for k, v in x.payload.items()}, x.seq_lens)
```

Now we can check `x.payload` and see features looks like real padded data:
```python
>>> x.payload['mcc_code']
Out: 
tensor([[8, 1, 0, 0, 0, 0, 0, 0],
        [5, 5, 9, 9, 4, 9, 3, 1],
        [4, 2, 2, 3, 3, 0, 0, 0]])
```

All invalid tokens are replaced with zeros.

Generally, all layers respect `PaddedBatch.seq_lens` and no explicit zeroing of padded characters is required.

## `ptls.nn.TrxEncoder`

Now we have an input data:
```python
x = PaddedBatch(
    payload={
        'mcc_code': torch.randint(1, 10, (3, 8)),
        'currency': torch.randint(1, 4, (3, 8)),
        'amount': torch.randn(3, 8) * 4 + 5,
    },
    length=torch.Tensor([2, 8, 5]).long()
)
```
And se can define a TrxEncoder
```python
model = TrxEncoder(
    embeddings={
        'mcc_code': {'in': 10, 'out': 6},
        'currency': {'in': 4, 'out': 2},
    },
    numeric_values={'amount': 'identity'},
)
```
We should provide feature description to `TrxEncoder`.
Dictionary size and embedding size for categorical features. Scaler name for numerical features.
`identity` means no rescaling.

`TrxEncoder` concatenate all feature embeddings, sow output embedding size will be `6 + 2 + 1`.
You may get output size from `TrxEncoder` with property:
```python
>>> model.output_size
Out[]: 6
```

Let's transform our features to embeddings
```python
z = model(x)
```

`z` is also `PaddedBatch`. `z.seq_lens` equals `x.seq_lens`.
`z.payload` isn't dict, it's tensor of shape (B, T, H). In our example `B, T = 3, 8` is input feature shape,
`H = 6` is output size of model.

Now we can use other layers which consume transactional embeddings.


## Classes
See docstrings for classes:

- `ptls.nn.PaddedBatch`
- `ptls.nn.TrxEncoder`