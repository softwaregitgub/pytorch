"""
        1、训练误差和泛化误差
            影响模型泛化的因素：
                1）可调整参数的数量。当可调整参数的数量（有时称为自由度）很大时，模型往往更容易过拟合。
                2）参数采用的值。当权重的取值范围较大时，模型可能更容易过拟合。
                3）训练样本的数量。即使模型很简单，也很容易过拟合只包含一两个样本的数据集。而过拟合一个有数百万个样本的数据集则需要一个极其灵活的模型。

"""

"""
        2、
        欠拟合（underfitting）
            我们的训练和验证误差之间的泛化误差很小， 我们有理由相信可以用一个更复杂的模型降低训练误差。 
        过拟合（overfitting）
            训练误差明显低于验证误差时
"""

