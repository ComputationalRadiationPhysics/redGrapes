/* Copyright 2020 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <cuda.h>
#include <pngwriter.h>
#include <iomanip>
#include <iostream>
#include <redGrapes/helpers/cuda/stream.hpp>
#include <redGrapes/manager.hpp>
#include <redGrapes/resource/fieldresource.hpp>
#include <redGrapes/resource/ioresource.hpp>

namespace rg = redGrapes;

struct Color
{
    float r, g, b;
};

__global__ void mandelbrot(double begin_x, double end_x, double begin_y, double end_y, int buffer_width, int buffer_height, Color * out)
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
    {
        out[index] = Color{cosf(float(i) / 7.0), cosf(2.0 + float(i) / 11.0), cosf(4.0 + float(i) / 13.0)};
    }
}

int main()
{
    rg::Manager<
        rg::TaskProperties<rg::ResourceProperty>,
        rg::ResourceEnqueuePolicy>
        mgr;

    double mid_x = 0.41820187155955555;
    double mid_y = 0.32743154895555555;

    size_t width = 4096;
    size_t height = 4096;
    size_t area = width * height;

    rg::helpers::cuda::StreamResource<decltype(mgr)> cuda_stream(mgr, 0);
    mgr.getScheduler().schedule[0].set_wait_hook([cuda_stream] { cuda_stream.poll(); });

    rg::IOResource<Color *> host_buffer;
    rg::IOResource<Color *> device_buffer;

    mgr.emplace_task(
        [area](auto host_buffer) {
            *host_buffer = new Color[area];
        },
        host_buffer.write());

    mgr.emplace_task(
        [area](auto device_buffer) {
            void * ptr;
            cudaMalloc(&ptr, area * sizeof(Color));
            *device_buffer = (Color *)ptr;
        },
        device_buffer.write());

    float w = 1.0;
    for(int i = 0; i < 200; ++i)
    {
        w *= 0.75;
        /*
         * calculate picture
         */
        mgr.emplace_task(
            [width, height, area, i, mid_x, mid_y, w](auto cuda_stream, auto device_buffer) {
                double begin_x = mid_x - w;
                double end_x   = mid_x + w;
                double begin_y = mid_y - w;
                double end_y   = mid_y + w;

                dim3 threadsPerBlock(8, 8);
                dim3 numBlocks(width / threadsPerBlock.x, height / threadsPerBlock.y);
                mandelbrot<<<numBlocks, threadsPerBlock, 0, cuda_stream>>>(
                    begin_x, end_x, begin_y, end_y, width, height, *device_buffer);
            },
            cuda_stream,
            device_buffer.write());

        /*
         * copy data
         */
        mgr.emplace_task(
            [area](auto cuda_stream, auto host_buffer, auto device_buffer) {
                cudaMemcpyAsync(*host_buffer, *device_buffer, area * sizeof(Color), cudaMemcpyDeviceToHost, cuda_stream);
                cuda_stream.sync();
            },
            cuda_stream,
            host_buffer.write(),
            device_buffer.read());

        /*
         * write png
         */
        int step = 0;
        mgr.emplace_task(
            [step, width, height, i](auto host_buffer) {
                std::stringstream step;
                step << std::setw(6) << std::setfill('0') << i;
                std::string filename("mandelbrot_" + step.str() + ".png");
                pngwriter png(width, height, 0, filename.c_str());
                png.setcompressionlevel(9);

                for(int y = 0; y < height; ++y)
                {
                    for(int x = 0; x < width; ++x)
                    {
                        auto & color = (*host_buffer)[x + y * width];
                        png.plot(x + 1, height - y, color.r, color.g, color.b);
                    }
                }

                png.close();
            },
            host_buffer.read());
    }

    /*
     * cleanup
     */
    mgr.emplace_task(
        [](auto host_buffer) {
            delete *host_buffer;
        },
        host_buffer.write());

    mgr.emplace_task(
        [](auto device_buffer) {
            cudaFree(*device_buffer);
        },
        device_buffer.write());
}
