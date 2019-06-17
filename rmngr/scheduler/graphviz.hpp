
#pragma once

#include <rmngr/scheduler/scheduler.hpp>

#include <boost/property_map/property_map.hpp>
#include <boost/property_map/function_property_map.hpp>

namespace rmngr
{

/**
 * Prints the Scheduling-Graph in a new file.
 * ColorProperty must be used in a SchedulingPolicy
 * and provide std::string color()
 */
template < typename ColorPolicy >
struct GraphvizWriter : DefaultSchedulingPolicy
{
    struct Property : DefaultSchedulingPolicy::Property
    {
        std::string label;
    };

    template <typename Graph>
    void update( Graph & graph, SchedulerInterface & scheduler )
    {
        static int step = 0;
        ++step;
        std::string name = std::string( "Step " ) + std::to_string( step );
        std::string path = std::string( "step_" ) + std::to_string( step ) +
                           std::string( ".dot" );
        std::cout << "write schedulinggraph to " << path << std::endl;
        std::ofstream file( path );

        graph.write_graphviz(
            file,
            boost::make_function_property_map< typename Graph::ID >(
                [&]( typename Graph::ID s ) {
		    std::string label;

		    // name
                    label += s->template property<GraphvizWriter>().label;
		    label += "\n";

		    // add resource information
		    std::ostringstream stream;
		    stream << s->template property<ResourceUserPolicy>();
		    label += stream.str();

                    return label;
                }
            ),
            boost::make_function_property_map< typename Graph::ID >(
                []( typename Graph::ID s ) {
                    return s->template property<ColorPolicy>().color();
                }
            ),
            name
        );

        file.close();
    }
};

}; // namespace rmngr

