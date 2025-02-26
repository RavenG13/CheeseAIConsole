Forward Pipline

类 forward{
	
	锁 输入锁
	锁 输出锁

	队列 输入缓冲区
	张量 输出缓冲区
	锁[] 输出锁

	放入(张量){
		输入锁.waitone()
		放入数据并获取位置
		锁释放
		结构体 输出(位置，锁，null)
		输出.锁.lock()
		输出锁.append(输出.锁)
		return 输出
	}

	前向函数()
	{
		获取输入锁
		输入缓冲区转换为张量
		释放输入锁

		输出缓冲区 = 神经网络forward(张量)
		for 锁 in 输出锁：
			锁.release()

	}
}
结构体 输出{
	int 位置
	锁 锁
	张量 输出
}

调用：
{
	结构体 输出 = 放入(张量)
	输出.waitone()
}
