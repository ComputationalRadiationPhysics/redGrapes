#pragma once

class ResourceAccess
{
    public:
        ResourceAccess(int ri, bool write_) : resourceID(ri), write(write_)
        {
        }

        int resourceID;
        bool write;
};

// check if a depends on b
bool check_dependency(ResourceAccess const& a, ResourceAccess const& b)
{
    return (a.resourceID == b.resourceID) && (b.write || a.write);
}

class Resource
{
    public:
        Resource(int ri)
            : read(ri, false), write(ri, true)
        {
        }

        ResourceAccess read;
        ResourceAccess write;
};

