
#include <catch/catch.hpp>

#include <redGrapes/task/task_space.hpp>

struct TestTask
{
    unsigned id;
    unsigned task_id;

    using VertexPtr = std::shared_ptr<redGrapes::PrecedenceGraphVertex<TestTask>>;
    using WeakVertexPtr = std::weak_ptr<redGrapes::PrecedenceGraphVertex<TestTask>>;
    
    /*
     * Create the following graph:
     *     0     2
     *    / \    |
     *   1   3   4
     *   |  / \ /
     *   6  5  7
     */
    static bool is_serial(TestTask const & a, TestTask const & b)
    {
        return (a.id == 0 && b.id == 1)
            || (a.id == 0 && b.id == 3)
            || (a.id == 2 && b.id == 4)
            || (a.id == 1 && b.id == 6)
            || (a.id == 3 && b.id == 5)
            || (a.id == 3 && b.id == 7)
            || (a.id == 4 && b.id == 7);
    }
};

TEST_CASE("precedence graph")
{
    /*
    auto precedence_graph = std::make_shared<redGrapes::PrecedenceGraph<TestTask, TestTask>>();
    auto task_space = std::make_shared<redGrapes::TaskSpace<TestTask>>(precedence_graph);

    for(unsigned id = 0; id < 8; ++id)
        task_space->push(std::make_unique<TestTask>(TestTask{id, 0}));

    while( auto vertex_ptr = task_space->next() )
    {
        unsigned taskid = (*vertex_ptr)->task->id;
        unsigned j = 0;
        for( auto in_ptr : (*vertex_ptr)->in_edges )
        {
            unsigned in_taskid = in_ptr.lock()->task->id;
            REQUIRE((
                (taskid == 1 && j == 0 && in_taskid == 0) ||
                (taskid == 3 && j == 0 && in_taskid == 0) ||
                (taskid == 4 && j == 0 && in_taskid == 2) ||
                (taskid == 5 && j == 0 && in_taskid == 3) ||
                (taskid == 6 && j == 0 && in_taskid == 1) ||
                (taskid == 7 && j == 0 && in_taskid == 4) ||
                (taskid == 7 && j == 1 && in_taskid == 3)
            ));
            j++;
        }
    }

    for( auto src_ptr : precedence_graph->tasks )
    {
        unsigned src_taskid = src_ptr->task->id;
        unsigned j = 0;
        for( auto dst_ptr : src_ptr->out_edges )
        {
            unsigned dst_taskid = dst_ptr.lock()->task->id;
            REQUIRE((
                (src_taskid == 0 && j == 0 && dst_taskid == 1) ||
                (src_taskid == 0 && j == 1 && dst_taskid == 3) ||
                (src_taskid == 1 && j == 0 && dst_taskid == 6) ||
                (src_taskid == 2 && j == 0 && dst_taskid == 4) ||
                (src_taskid == 3 && j == 0 && dst_taskid == 5) ||
                (src_taskid == 3 && j == 1 && dst_taskid == 7) ||
                (src_taskid == 4 && j == 0 && dst_taskid == 7)
            ));
            j++;
        }
    }
    */
}

