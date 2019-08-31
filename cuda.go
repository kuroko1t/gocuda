// Copyright 2019 kurosawa. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

package cu

// #include "/usr/local/cuda/include/cuda.h"
// #cgo LDFLAGS: -L/usr/local/cuda/lib64/ -lcuda
import "C"
import (
	"fmt"
	"os"
	"os/exec"
	"path"
	"unsafe"
)

// Based on cuda driver API

// Gets the string description of an error code
func GetErrorString(error CUresult) (string, CUresult) {
	var cpstr *C.char
	status := C.cuGetErrorString(C.CUresult(error), &cpstr)
	pstr := C.GoString(cpstr)
	return pstr, CUresult(status)
}

// Gets the string representation of an error code enum name
func GetErrorName(error CUresult) (string, CUresult) {
	var cpstr *C.char
	status := C.cuGetErrorName(C.CUresult(error), &cpstr)
	pstr := C.GoString(cpstr)
	return pstr, CUresult(status)
}

// Returns information about the device
func DeviceGetAttribute(attrib CUdevice_attribute, dev CUdevice) (int, CUresult) {
	var pi C.int
	status := C.cuDeviceGetAttribute(&pi, C.CUdevice_attribute(attrib), C.CUdevice(dev))
	return int(pi), CUresult(status)
}

// Initialize the CUDA driver API
func Init() CUresult {
	return CUresult(C.cuInit(0))
}

// Returns the latest CUDA version supported by driver
func DriverGetVersion() (int, CUresult) {
	var driverVersion C.int
	status := C.cuDriverGetVersion(&driverVersion)
	return int(driverVersion), CUresult(status)
}

// Returns a handle to a compute device
func DeviceGet(ordinal int) (CUdevice, CUresult) {
	var cdevice C.CUdevice
	status := C.cuDeviceGet(&cdevice, C.int(ordinal))
	device := CUdevice(cdevice)
	return device, CUresult(status)
}

// Returns the number of compute-capable devices
func DeviceGetCount() (int, CUresult) {
	var count C.int
	status := C.cuDeviceGetCount(&count)
	return int(count), CUresult(status)
}

// Create a CUDA context
func CtxCreate(flags uint32, dev CUdevice) (CUcontext, CUresult) {
	var context C.CUcontext
	status := C.cuCtxCreate(&context, C.uint(flags), C.CUdevice(dev))
	return CUcontext(context), CUresult(status)
}

// Destroy a CUDA context
func CtxDestroy(ctx CUcontext) CUresult {
	status := C.cuCtxDestroy(C.CUcontext(ctx))
	return CUresult(status)
}

// Block for a context's tasks to complete
func CtxSynchronize() CUresult {
	status := C.cuCtxSynchronize()
	return CUresult(status)
}

// Returns the device ID for the current context
func CtxGetDevice() (CUdevice, CUresult) {
	var device C.CUdevice
	status := C.cuCtxGetDevice(&device)
	return CUdevice(device), CUresult(status)
}

// Allocates device memory
func MemAlloc(bytesize uint32) (CUdeviceptr, CUresult) {
	var dptr C.CUdeviceptr
	status := C.cuMemAlloc(&dptr, C.ulong(bytesize))
	return CUdeviceptr(dptr), CUresult(status)
}

// Allocates memory that will be automatically managed by the Unified Memory system
func MemAllocManaged(bytesize, flags uint32) (CUdeviceptr, CUresult) {
	var dptr C.CUdeviceptr
	status := C.cuMemAllocManaged(&dptr, C.ulong(bytesize), C.uint(flags))
	return CUdeviceptr(dptr), CUresult(status)
}

// Copies memory from Host to Device
func MemcpyHtoD(dstDevice CUdeviceptr, srcHost interface{}, ByteCount uint32) CUresult {
	var status C.CUresult
	switch srcHostValue := srcHost.(type) {
	case *int:
		status = C.cuMemcpyHtoD(C.CUdeviceptr(dstDevice), unsafe.Pointer(srcHostValue), C.size_t(ByteCount))
	case *float32:
		status = C.cuMemcpyHtoD(C.CUdeviceptr(dstDevice), unsafe.Pointer(srcHostValue), C.size_t(ByteCount))
	case *float64:
		status = C.cuMemcpyHtoD(C.CUdeviceptr(dstDevice), unsafe.Pointer(srcHostValue), C.size_t(ByteCount))
	default:
		break
	}
	return CUresult(status)
}

// Copies memory from Device to Host
func MemcpyDtoH(dstHost interface{}, srcDevice CUdeviceptr, ByteCount uint32) CUresult {
	var status C.CUresult
	switch dstHostValue := dstHost.(type) {
	case *int:
		status = C.cuMemcpyDtoH(unsafe.Pointer(dstHostValue), C.CUdeviceptr(srcDevice), C.size_t(ByteCount))
	case *float32:
		status = C.cuMemcpyDtoH(unsafe.Pointer(dstHostValue), C.CUdeviceptr(srcDevice), C.size_t(ByteCount))
	case *float64:
		status = C.cuMemcpyDtoH(unsafe.Pointer(dstHostValue), C.CUdeviceptr(srcDevice), C.size_t(ByteCount))
	default:
		break
	}
	return CUresult(status)
}

// Copies memory
func Memcpy(dst, src CUdeviceptr, ByteCount uint32) CUresult {
	status := C.cuMemcpy(C.CUdeviceptr(dst), C.CUdeviceptr(src), C.size_t(ByteCount))
	return CUresult(status)
}

// Copies memory from Host to Device
func MemcpyHtoDAsync(dstDevice CUdeviceptr, srcHost interface{}, ByteCount uint32, hStream CUstream) CUresult {
	var status C.CUresult
	switch srcHostValue := srcHost.(type) {
	case *int:
		status = C.cuMemcpyHtoDAsync(C.CUdeviceptr(dstDevice), unsafe.Pointer(srcHostValue), C.size_t(ByteCount), C.CUstream(hStream))
	case *float32:
		status = C.cuMemcpyHtoDAsync(C.CUdeviceptr(dstDevice), unsafe.Pointer(srcHostValue), C.size_t(ByteCount), C.CUstream(hStream))
	case *float64:
		status = C.cuMemcpyHtoDAsync(C.CUdeviceptr(dstDevice), unsafe.Pointer(srcHostValue), C.size_t(ByteCount), C.CUstream(hStream))
	default:
		break
	}
	return CUresult(status)
}

// Gets free and total memory
func MemGetInfo() (uint32, uint32, CUresult) {
	var free C.ulong
	var total C.ulong
	status := C.cuMemGetInfo(&free, &total)
	return uint32(free), uint32(total), CUresult(status)
}

// Copies memory from Device to Host
func MemcpyDtoHAsync(dstHost interface{}, srcDevice CUdeviceptr, ByteCount uint32, hStream CUstream) CUresult {
	var status C.CUresult
	switch dstHostValue := dstHost.(type) {
	case *int:
		status = C.cuMemcpyDtoHAsync(unsafe.Pointer(dstHostValue), C.CUdeviceptr(srcDevice), C.size_t(ByteCount), C.CUstream(hStream))
	case *float32:
		status = C.cuMemcpyDtoHAsync(unsafe.Pointer(dstHostValue), C.CUdeviceptr(srcDevice), C.size_t(ByteCount), C.CUstream(hStream))
	case *float64:
		status = C.cuMemcpyDtoHAsync(unsafe.Pointer(dstHostValue), C.CUdeviceptr(srcDevice), C.size_t(ByteCount), C.CUstream(hStream))
	default:
		break
	}
	return CUresult(status)
}

//Copies memory from Device to Device
func MemcpyDtoDAsync(dstDevice, srcDevice CUdeviceptr, ByteCount uint32, hStream CUstream) CUresult {
	var status C.CUresult
	status = C.cuMemcpyDtoDAsync(C.CUdeviceptr(dstDevice), C.CUdeviceptr(srcDevice), C.size_t(ByteCount), C.CUstream(hStream))
	return CUresult(status)
}

// Copies memory asynchronously
func MemcpyAsync(dst, src CUdeviceptr, ByteCount uint32, hstream CUstream) CUresult {
	status := C.cuMemcpyAsync(C.CUdeviceptr(dst), C.CUdeviceptr(src), C.size_t(ByteCount), C.CUstream(hstream))
	return CUresult(status)
}

// Copies device memory between two contexts
func MemcpyPeer(dstDevice CUdeviceptr, dstContext CUcontext, srcDevice CUdeviceptr, srcContext CUcontext, ByteCount uint32) CUresult {
	status := C.cuMemcpyPeer(C.CUdeviceptr(dstDevice), C.CUcontext(dstContext), C.CUdeviceptr(srcDevice), C.CUcontext(srcContext), C.size_t(ByteCount))
	return CUresult(status)
}

// Copies device memory between two contexts asynchronously.
func MemcpyPeerAsync(dstDevice CUdeviceptr, dstContext CUcontext, srcDevice CUdeviceptr, srcContext CUcontext, ByteCount uint32, hStream CUstream) CUresult {
	status := C.cuMemcpyPeerAsync(C.CUdeviceptr(dstDevice), C.CUcontext(dstContext), C.CUdeviceptr(srcDevice), C.CUcontext(srcContext), C.size_t(ByteCount), C.CUstream(hStream))
	return CUresult(status)
}

// Loads a compute module
func ModuleLoad(fname string) (CUmodule, CUresult) {
	var module C.CUmodule
	status := C.cuModuleLoad(&module, C.CString(fname))
	return CUmodule(module), CUresult(status)
}

// Returns a function handle
func ModuleGetFunction(hmod CUmodule, name string) (CUfunction, CUresult) {
	var hfunc C.CUfunction
	status := C.cuModuleGetFunction(&hfunc, C.CUmodule(hmod), C.CString(name))
	return CUfunction(hfunc), CUresult(status)
}

// Launches a CUDA function
func LaunchKernel(f CUfunction, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, shredMemBytes uint32, hStream CUstream, kernelParams, extra []*CUdeviceptr) CUresult {
	kernelv := C.malloc(C.size_t(len(kernelParams) * int(unsafe.Sizeof(kernelParams[0]))))
	kernelp := C.malloc(C.size_t(len(extra) * int(unsafe.Sizeof(kernelParams[0]))))
	defer C.free(kernelv)
	defer C.free(kernelp)
	for i := range kernelParams {
		*((*unsafe.Pointer)(offset(kernelp, i))) = offset(kernelv, i)
		*((*uint64)(offset(kernelv, i))) = *((*uint64)(kernelParams[i]))
	}
	status := C.cuLaunchKernel(f, C.uint(gridDimX), C.uint(gridDimY), C.uint(gridDimZ),
		C.uint(blockDimX), C.uint(blockDimY), C.uint(blockDimZ), C.uint(shredMemBytes),
		C.CUstream(hStream), (*unsafe.Pointer)(kernelp), nil)
	return CUresult(status)
}

// Unloads a module
func ModuleUnLoad(hmod CUmodule) CUresult {
	status := C.cuModuleUnload(C.CUmodule(hmod))
	return CUresult(status)
}

// Frees device memory
func MemFree(dptr CUdeviceptr) CUresult {
	status := C.cuMemFree(C.CUdeviceptr(dptr))
	return CUresult(status)
}

// Returns an identifer string for the device
func DeviceGetName(len int, dev CUdevice) (string, CUresult) {
	var name C.char
	status := C.cuDeviceGetName(&name, C.int(len), C.CUdevice(dev))
	return C.GoString(&name), CUresult(status)
}

// Returns the total amount of memory on the device
func DeviceTotalMem(dev CUdevice) (int, CUresult) {
	var bytes C.size_t
	status := C.cuDeviceTotalMem(&bytes, C.CUdevice(dev))
	return int(bytes), CUresult(status)
}

// Create a stream
func StreamCreate(flags uint32) (CUstream, CUresult) {
	var phStream C.CUstream
	status := C.cuStreamCreate(&phStream, C.uint(flags))
	return CUstream(phStream), CUresult(status)
}

// Determine status of a compute stream
func StreamQuery(hStream CUstream) CUresult {
	status := C.cuStreamQuery(hStream)
	return CUresult(status)
}

// Wait until a stream's tasks are completed
func StreamSynchronize(hStream CUstream) CUresult {
	status := C.cuStreamSynchronize(C.CUstream(hStream))
	return CUresult(status)
}

// Destroys a stream
func StreamDestroy(hStream CUstream) CUresult {
	status := C.cuStreamDestroy(C.CUstream(hStream))
	return CUresult(status)
}

// Return an UUID for the device
func DeviceGetUuid(dev CUdevice) (CUuuid, CUresult) {
	var uuid C.CUuuid
	status := C.cuDeviceGetUuid(&uuid, C.CUdevice(dev))
	return CUuuid(uuid), CUresult(status)
}

// Query the context associated with a stream
func StreamGetCtx(hStream CUstream) (CUcontext, CUresult) {
	var pctx C.CUcontext
	status := C.cuStreamGetCtx(C.CUstream(hStream), &pctx)
	return CUcontext(pctx), CUresult(status)
}

// Gets the context's API version.
func CtxGetApiVersion(ctx CUcontext) (int, CUresult) {
	var version C.uint
	status := C.cuCtxGetApiVersion(C.CUcontext(ctx), &version)
	return int(version), CUresult(status)
}

// gocuda original API

func Setup(deviceID int) CUresult {
	status := Init()
	if status != C.CUDA_SUCCESS {
		return status
	}
	ctx, status := CtxCreate(0, CUdevice(deviceID))
	cudesc := CuDesc{}
	cudesc.context = ctx
	cuDescMap[0] = cudesc
	return status
}

func Teardown(deviceID int) CUresult {
	status := CtxDestroy(cuDescMap[CUdevice(deviceID)].context)
	delete(cuDescMap, 0)
	return CUresult(status)
}

func CreateModuleFromFile(file string) (CUmodule, CUresult) {
	cmd := exec.Command("nvcc", "--ptx", file)
	if err := cmd.Run(); err != nil {
		fmt.Println(err)
		return nil, CUresult(CUDA_ERROR_FILE_NOT_FOUND)
	}
	_, fileBase := path.Split(file)
	fileBase = fileBase[:len(fileBase)-len(path.Ext(fileBase))]
	module, status := ModuleLoad(fileBase + ".ptx")
	if err := os.Remove(fileBase + ".ptx"); err != nil {
		fmt.Println(err)
		return nil, CUresult(CUDA_ERROR_FILE_NOT_FOUND)
	}
	return module, CUresult(status)
}

func offset(ptr unsafe.Pointer, i int) unsafe.Pointer {
	return unsafe.Pointer(uintptr(ptr) + unsafe.Sizeof(ptr)*uintptr(i))
}
