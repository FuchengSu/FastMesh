#// CMake 最低版本号要求
cmake_minimum_required (VERSION 2.8)
#// 项目信息
project (Demo3)
#// 查找当前目录下的所有源文件
set(CMAKE_CXX_STANDARD 14)
#// OpenCV
#//当有多个opencv版本时，可指定优先查找opencv的目录OpenCV_DIR
find_package(OpenCV 4.5.4 REQUIRED)

find_package(Boost REQUIRED COMPONENTS


filesystem   # 我的工程中只使用了 boost 的 filesystem 功能,因此这里只有一个组件
)
if(NOT Boost_FOUND)
    message("Not found Boost")
endif()

find_package(Eigen3)

#// 添加include文件查找目录，这里把CMakeLists.txt所在目录添加了进去
include_directories(
	${OpenCV_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    "/usr/local/include/"
)
#// 指定生成目标 
add_executable(Demo computeK.cpp)
#// 添加链接库
target_link_libraries(Demo 
    ${OpenCV_LIBS}
    ${Boost_LIBRARIES}
)