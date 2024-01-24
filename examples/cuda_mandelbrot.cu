/* Copyright 2020 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <cuda.h>
#include <pngwriter.h>

#include <chrono>
#include <functional>
#include <iomanip>
#include <iostream>

enum SchedulerTag
{
    SCHED_CUDA
};

#define REDGRAPES_TASK_PROPERTIES dispatch::cuda::CudaTaskProperties, scheduler::SchedulingTagProperties<SchedulerTag>

#include <redGrapes/dispatch/cuda/scheduler.hpp>
#include <redGrapes/redGrapes.hpp>
#include <redGrapes/resource/fieldresource.hpp>
#include <redGrapes/resource/ioresource.hpp>
#include <redGrapes/scheduler/default_scheduler.hpp>
#include <redGrapes/scheduler/tag_match.hpp>
#include <redGrapes/task/property/resource.hpp>

namespace rg = redGrapes;

struct Color
{
    float r, g, b;
};

__global__ void hello_world()
{
}

__global__ void mandelbrot(
    double begin_x,
    double end_x,
    double begin_y,
    double end_y,
    int buffer_width,
    int buffer_height,
    Color* out)
{
    int xi = blockIdx.x * blockDim.x + threadIdx.x;
    int yi = blockIdx.y * blockDim.y + threadIdx.y;
    int index = xi + yi * buffer_width;
    double xf = begin_x + (end_x - begin_x) * double(xi) / double(buffer_width);
    double yf = begin_y + (end_y - begin_y) * double(yi) / double(buffer_height);

    double z_re = 0.0;
    double z_im = 0.0;
    int i;
    for(i = 0; i < 1000 && (z_re * z_re + z_im * z_im) < 4; ++i)
    {
        double new_z_re = z_re * z_re - z_im * z_im + xf;
        z_im = 2 * z_re * z_im + yf;
        z_re = new_z_re;
    }

    if(i == 1000)
        out[index] = Color{0.0, 0.0, 0.0};
    else
        out[index] = Color{cosf(float(i) / 7.0), cosf(2.0 + float(i) / 11.0), cosf(4.0 + float(i) / 13.0)};
}

int main()
{
    auto default_scheduler = std::make_shared<rg::scheduler::DefaultScheduler>(4 /* number of CPU workers */);

    auto cuda_scheduler = std::make_shared<rg::dispatch::cuda::CudaScheduler>(
        [](rg::Task const& t) { return t.required_scheduler_tags.test(SCHED_CUDA); },
        4 /* number of cuda streams */
    );

    rg::idle = [cuda_scheduler] { cuda_scheduler->poll(); };

    rg::init(rg::scheduler::make_tag_match_scheduler().add({}, default_scheduler).add({SCHED_CUDA}, cuda_scheduler));

    double mid_x = 0.41820187155955555;
    double mid_y = 0.32743154895555555;

    size_t width = 4096;
    size_t height = 4096;
    size_t area = width * height;

    rg::IOResource<Color*> host_buffer;
    rg::IOResource<Color*> device_buffer;

    rg::emplace_task(
        [area](auto host_buffer)
        {
            void* ptr;
            cudaMallocHost(&ptr, area * sizeof(Color));
            *host_buffer = (Color*) ptr;
        },
        host_buffer.write());

    rg::emplace_task(
        [area](auto device_buffer)
        {
            void* ptr;
            cudaMalloc(&ptr, area * sizeof(Color));
            *device_buffer = (Color*) ptr;
        },
        device_buffer.write());

    // warmup cuda
    // hello_world<<< 1, 1, 0, 0 >>>();
    // cudaMemcpy(*host_buffer, *device_buffer, sizeof(Color), cudaMemcpyDeviceToHost);

    auto t1 = std::chrono::high_resolution_clock::now();

    float w = 1.0;
    for(int i = 0; i < 10; ++i)
    {
        w *= 0.75;
        /*
         * calculate picture
         */
        rg::emplace_task(
            [width, height, area, i, mid_x, mid_y, w](auto device_buffer)
            {
                double begin_x = mid_x - w;
                double end_x = mid_x + w;
                double begin_y = mid_y - w;
                double end_y = mid_y + w;

                dim3 threadsPerBlock(8, 8);
                dim3 numBlocks(width / threadsPerBlock.x, height / threadsPerBlock.y);

                mandelbrot<<<numBlocks, threadsPerBlock, 0, rg::dispatch::cuda::current_stream>>>(
                    begin_x,
                    end_x,
                    begin_y,
                    end_y,
                    width,
                    height,
                    *device_buffer);
                std::cout << "launched kernel to stream " << rg::dispatch::cuda::current_stream << std::endl;
            },
            rg::TaskProperties::Builder().scheduling_tags({SCHED_CUDA}),
            device_buffer.write());

        /*
         * copy data
         */
        rg::emplace_task(
            [area](auto host_buffer, auto device_buffer)
            {
                cudaMemcpyAsync(
                    *host_buffer,
                    *device_buffer,
                    area * sizeof(Color),
                    cudaMemcpyDeviceToHost,
                    rg::dispatch::cuda::current_stream);
                std::cout << "launched memcpy to stream " << rg::dispatch::cuda::current_stream << std::endl;
            },
            rg::TaskProperties::Builder().scheduling_tags({SCHED_CUDA}),
            host_buffer.write(),
            device_buffer.read());

        /*
         * write png
         */
        rg::emplace_task(
            [width, height, i](auto host_buffer)
            {
                std::stringstream step;
                step << std::setw(6) << std::setfill('0') << i;

                std::string filename("mandelbrot_" + step.str() + ".png");
                pngwriter png(width, height, 0, filename.c_str());
                png.setcompressionlevel(9);

                for(int y = 0; y < height; ++y)
                {
                    for(int x = 0; x < width; ++x)
                    {
                        auto& color = (*host_buffer)[x + y * width];
                        png.plot(x + 1, height - y, color.r, color.g, color.b);
                    }
                }

                png.close();
                std::cout << "wrote png" << std::endl;
            },
            host_buffer.read());
    }

    rg::emplace_task([](auto b) {}, host_buffer.write()).get();

    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "runtime: " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << " μs"
              << std::endl;

    /*
     * cleanup
     */
    rg::emplace_task([](auto host_buffer) { cudaFreeHost(*host_buffer); }, host_buffer.write());

    rg::emplace_task([](auto device_buffer) { cudaFree(*device_buffer); }, device_buffer.write());
}
