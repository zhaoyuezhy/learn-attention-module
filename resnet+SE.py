 import tensorflow as tf
 
# 暂时的
import



def identity_block(input_xs, out_dim, with_shortcut_conv_BN=False):
    if with_shortcut_conv_BN:
        pass
    else:
        #返回与input的形状和内容均相同的张量，即shortcut等同于input_xs
        shortcut = tf.identity(input_xs)
    #input输入的channel数
    input_channel = input_xs.get_shape().as_list()[-1]
    #如果输入的channel数不等于输出的channel数的话
    if input_channel != out_dim:
        #求输出的channel数减去输入的channel数的绝对值，作为pad填充值
        pad_shape = tf.abs(out_dim - input_channel)
        #name="padding"表示给该填充操作赋予名称为"padding"。使用了默认参数mode='CONSTANT'和constant_values=0，表示填充默认值0。
        #第二个参数为paddings填充的形状：即分别的批量维度、高、宽的维度上都不作填充，在channel维度上填充pad_shape//2的数量。
        shortcut = tf.pad(shortcut, [[0, 0], [0, 0], [0, 0], [pad_shape // 2, pad_shape // 2]], name="padding")
    #残差卷积块中的3个Conv2D卷积的卷积核大小分别为1x1、3x3、1x1
    conv = tf.keras.layers.Conv2D(filters=out_dim // 4, kernel_size=1, padding="SAME", activation=tf.nn.relu)(input_xs)
    conv = tf.keras.layers.BatchNormalization()(conv)
    conv = tf.keras.layers.Conv2D(filters=out_dim // 4, kernel_size=3, padding="SAME", activation=tf.nn.relu)(conv)
    conv = tf.keras.layers.BatchNormalization()(conv)
    conv = tf.keras.layers.Conv2D(filters=out_dim // 4, kernel_size=1, padding="SAME", activation=tf.nn.relu)(conv)
    conv = tf.keras.layers.BatchNormalization()(conv)
    #下面开始加载SENet模块
    #返回的为[批量维度、高、宽、channel维度]
    shape = conv.get_shape().as_list()
    #默认参数为keepdims=False的话，不会再保留运算所在的维度。设置keepdims=True的话，会保留运算所在的维度为1。
    #[批量维度、高、宽、channel维度]经过reduce_mean后转换为[批量维度、channel维度]
    se_module = tf.reduce_mean(conv, [1, 2])
    #第一个Dense：shape[-1]/reduction_ratio：即把input_channel再除以reduction_ratio，使channel下降到指定维度数
    se_module = tf.keras.layers.Dense(shape[-1] / 16, activation=tf.nn.relu)(se_module)
    #第二个Dense：重新回升到与input_channel相同的原始维度数
    se_module = tf.keras.layers.Dense(shape[-1], activation=tf.nn.relu)(se_module)
    se_module = tf.nn.sigmoid(se_module)
    #把[批量维度、channel维度]重新转换为[批量维度、高、宽、channel维度]，即[批量维度、1、1、channel维度]
    se_module = tf.reshape(se_module, [-1, 1, 1, shape[-1]])
    #multiply元素乘法：SENet模块输出值se_module 和 残差卷积输出conv(即SENet模块输入值conv)
    se_module = tf.multiply(conv, se_module)
    #残差连接：对残差的原始输入shortcut(即input_xs) 与 SENet模块输出值se_module 进行求和
    output_ys = tf.add(shortcut, se_module)
    output_ys = tf.nn.relu(output_ys)
    return output_ys