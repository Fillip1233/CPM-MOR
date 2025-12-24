from neuralop.models import FNO
if __name__ == "__main__":
    from neuralop.models import FNO
    model1 =   FNO(n_modes=(64,),
            hidden_channels=64,
            in_channels=10000,
            out_channels=10000)
    total_params = sum(p.numel() for p in model1.parameters())
    print(f"Total number of parameters: {total_params}")
    # 测试代码
    pass