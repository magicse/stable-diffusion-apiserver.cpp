cmake_minimum_required(VERSION 3.13)
project(sd_server)


# Windows версия	_WIN32_WINNT значение
# Windows XP	0x0501
# Windows Vista	0x0600
# Windows 7	0x0601
# Windows 8	0x0602
# Windows 8.1	0x0603
# Windows 10	0x0A00
# Windows 11	0x0A00

macro(get_WIN32_WINNT version)
    if (WIN32 AND CMAKE_SYSTEM_VERSION)
        set(ver ${CMAKE_SYSTEM_VERSION})
        message(STATUS "MinGW detected, ver is ${ver}")
        string(REGEX MATCH "^([0-9]+).([0-9])" ver ${ver})
        string(REPLACE "." "" ver ${ver})
        message(STATUS "MinGW detected, ver is ${ver}")
        string(REGEX REPLACE "([0-9])" "0\\1" ver ${ver})
        message(STATUS "MinGW detected, ver is ${ver}")
        
        #string(REGEX MATCH "^([0-9]+).([0-9])" ver ${ver})
        #string(REGEX MATCH "^([0-9]+)" verMajor ${ver})

        set(${version} "0x${ver}")
    endif()
endmacro()

get_WIN32_WINNT(WIN_VER)
message(STATUS "_WIN32_WINNT will be set to ${WIN_VER}")

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Определяем платформу
if(WIN32)
    set(PLATFORM_LIBS ws2_32)
    if(MINGW)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libgcc -static-libstdc++")
    endif()
elseif(UNIX)
    set(PLATFORM_LIBS pthread)
    if(APPLE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
    endif()
endif()

# Устанавливаем платформенный макрос для Windows 8
if(WIN32)
    # add_definitions(-D_WIN32_WINNT=0x0601)
	add_definitions(-D_WIN32_WINNT=${WIN_VER})
endif()


set(HTTPLIB_HEADER_PATH "${CMAKE_CURRENT_SOURCE_DIR}/httplib/httplib.h")

# Add submodule directory for stable-diffusion
add_subdirectory(stable-diffusion.cpp)


# Include directories for stable-diffusion and its dependencies
include_directories(stable-diffusion.cpp src httplib)

# Create executable from your main.cpp
add_executable(sd_server sd_server.cpp)

# Link with the stable-diffusion library
target_link_libraries(sd_server stable-diffusion ${PLATFORM_LIBS})