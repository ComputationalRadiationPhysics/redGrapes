#pragma once

#include <rmngr/dependency_manager.hpp>

namespace rmngr
{

struct DefaultAccessProperty
{
};

template <typename ResourceAccessProperty>
class ResourceBase
{
    public:
        ResourceBase()
        {
        }

        class ResourceAccess
        {
            public:
                ResourceAccess(ResourceBase<ResourceAccessProperty>* const resource_, ResourceAccessProperty prop_)
                    : resource(resource_), prop(prop_)
                {
                }

                operator ResourceAccessProperty() const
                {
                    return this->prop;
                }

                ResourceAccessProperty operator() (void) const
                {
                    return this->prop;
                }

                ResourceAccessProperty& operator() (void)
                {
                    return this->prop;
                }

                bool check_dependency(ResourceAccess const& a) const
                {
                    if(this->resource == a.resource)
                        return this->resource->check_dependency(*this, a);

                    return false;
                }

                bool operator== (ResourceAccess const& a) const
                {
                    return (this->resource == a.resource && this->prop == a.prop);
                }

            private:
                ResourceBase<ResourceAccessProperty>* const resource;
                ResourceAccessProperty prop;
        }; // struct ResourceAccess

        struct ResourceAccessHasher
        {
            std::size_t operator() (ResourceAccess const& ra) const
            {
                return std::hash<ResourceBase<ResourceAccessProperty>*>(ra.resource);
            }
        }; // struct ResourceAccessHasher


        ResourceAccess create_resource_access(ResourceAccessProperty const& prop)
        {
            return ResourceAccess(this, prop);
        }

    private:
        virtual bool check_dependency(ResourceAccess const& a, ResourceAccess const& b)
        {
            return true;
        }
}; // class ResourceBase

template <typename AccessType, typename Dependency=BoolDependency, typename ResourceAccessProperty=DefaultAccessProperty>
class StaticResource : public ResourceBase<std::pair<ResourceAccessProperty, typename AccessType::Id> >
{
    public:
        using Id = typename AccessType::Id;
        using typename ResourceBase<std::pair<ResourceAccessProperty, Id> >::ResourceAccess;
        StaticResource()
        {
            if(! init)
            {
                AccessType::build_dependencies(access_dep);
                init = true;
            }
        }

        using ResourceBase<std::pair<ResourceAccessProperty, Id> >::create_resource_access;
        ResourceAccess create_resource_access(Id const& id)
        {
            return this->create_resource_access(std::make_pair(ResourceAccessProperty(), id));
        }

        static DependencyManager<Id, Dependency> access_dep;
        static bool init;

    private:
        bool check_dependency(ResourceAccess const& a, ResourceAccess const& b)
        {
            return access_dep.check_dependency(a().second, b().second);
        }
}; // class StaticResource

template<typename AccessType, typename Dependency, typename ResourceAccessProperty>
DependencyManager<typename AccessType::Id, Dependency> StaticResource<AccessType, Dependency, ResourceAccessProperty>::access_dep;
template<typename AccessType, typename Dependency, typename ResourceAccessProperty>
bool StaticResource<AccessType, Dependency, ResourceAccessProperty>::init = false;

} // namespace rmngr

