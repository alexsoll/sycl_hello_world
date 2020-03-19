#include <iostream>
#include <vector>
#include <CL/sycl.hpp>

void print_sep(std::string c) {
    for (int i = 0; i < 100; i++) {
        std::cout << c;
    }
    std::cout << "" << std::endl;
}

int main(int, char**) {

    const int size = 32000;

    std::array<cl::sycl::float4, size> a;
    std::array<cl::sycl::float4, size> b;
    std::array<cl::sycl::float4, size> c;
    std::cout << "123" << std::endl;

    for (size_t i = 0; i < size; i++) {
        a[i] = float(rand() % 100);
        b[i] = float(rand() % 100);
        c[i] = float(rand() % 100);
    }

    class MyDeviceSelector : public cl::sycl::device_selector {
        public:
            int operator()(const cl::sycl::device& Device) const override {
                using namespace cl::sycl::info;
                const std::string DeviceName = Device.get_info<device::name>();
                const std::string DeviceVendor = Device.get_info<device::vendor>();
                std::cout << DeviceName << " " << DeviceVendor << std::endl;
                return Device.is_gpu();
            }
    };

    print_sep("#");
    std::cout << "#" << "\t\t\t\tGet devices list..." << std::endl;
    print_sep("#");
    MyDeviceSelector Selector;
    try {
        cl::sycl::queue Queue(Selector);
        std::cout << "\nRunning on "
                << Queue.get_device().get_info<cl::sycl::info::device::name>()
                << "\n";
        std::cout << "Max work group size : "
                << Queue.get_device().get_info<cl::sycl::info::device::max_work_group_size>() 
                << std::endl;
        {
            cl::sycl::buffer<cl::sycl::float4, 1> a_sycl(a.data(), cl::sycl::range<1>(size));
            cl::sycl::buffer<cl::sycl::float4, 1> b_sycl(b.data(), cl::sycl::range<1>(size));
            cl::sycl::buffer<cl::sycl::float4, 1> c_sycl(c.data(), cl::sycl::range<1>(size));
            print_sep("#");
            std::cout << "#" << "\t\t\t\tRunning the application..." << std::endl;
            print_sep("#");
            Queue.submit([&] (cl::sycl::handler& cgh) {
                cl::sycl::stream kernelout(size, 256, cgh);
                auto a_acc = a_sycl.get_access<cl::sycl::access::mode::read>(cgh);
                auto b_acc = b_sycl.get_access<cl::sycl::access::mode::read>(cgh);
                auto c_acc = c_sycl.get_access<cl::sycl::access::mode::write>(cgh);
                cgh.parallel_for<class add_vectors>(sycl::range<1>(size), [=](cl::sycl::nd_item<1> item) {
                    int wiID = item.get_global_id()[0];
                    int grID = item.get_group()[0];
                    kernelout << "Global id - " << wiID << " ; " << "Group id - " << grID << cl::sycl::endl;
                    c_acc[wiID] = a_acc[wiID] + b_acc[wiID];
                });
            });
            Queue.wait_and_throw();
        }
    } catch (cl::sycl::exception const& e) {
        std::cout << "Caught asynchronous SYCL exception:\n"
          << e.what() << std::endl;
    }

    for (size_t i = 0; i < size; i++) {
        if ((a[i].x() + b[i].x()) != c[i].x()) {
            std::cout << "ERROR: The addition was performed incorrectly!" << std::endl;
            return 1;
        }
    }
    std::cout << "No errors" << std::endl;
    return 0;
}