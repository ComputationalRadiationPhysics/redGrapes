
#include <redGrapes/sync/cv.hpp>

#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <thread>

TEST_CASE("CV")
{
    for(int i = 0; i < 5000; ++i)
    {
        std::atomic<bool> finished = {false};

        redGrapes::CondVar cv;

        std::thread t(
            [&]
            {
                /* wait should run through without waiting,
                 * because notify came before wait
                 */
                cv.wait();
                finished = true;
            });

        cv.notify();

        auto end = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while(std::chrono::steady_clock::now() < end)
            if(finished)
                break;

        REQUIRE(finished);
        t.join();
    }
}
