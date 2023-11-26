import numpy as np
import pyopencl as cl
import matplotlib.pyplot as plt

# Исходный код ядра OpenCL для вычисления синусоиды
kernel_code = """
__kernel void sine_wave(__global float* input, __global float* output, const unsigned int n)
{
    int i = get_global_id(0);
    if (i < n)
        output[i] = sin(input[i]);
}
"""

def main():
    # Инициализация OpenCL
    platform = cl.get_platforms()[0]
    device = platform.get_devices()[0]
    context = cl.Context([device])
    queue = cl.CommandQueue(context)

    # Компиляция ядра OpenCL
    program = cl.Program(context, kernel_code).build()

    # Создание входных данных
    n = 1000
    input_data = np.linspace(0, 4 * np.pi, n).astype(np.float32)

    # Выделение буферов OpenCL
    input_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=input_data)
    output_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, input_data.nbytes)

    # Выполнение OpenCL ядра
    program.sine_wave(queue, input_data.shape, None, input_buffer, output_buffer, np.uint32(n))
    output_data = np.empty_like(input_data)
    cl.enqueue_copy(queue, output_data, output_buffer).wait()

    # Отрисовка графика
    plt.plot(output_data, label='Sine Wave')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
