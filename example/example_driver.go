package main

import (
	"fmt"
	"github.com/kuroko1t/gocuda"
	"unsafe"
)

func errCheck(status cu.CUresult) {
	if status != cu.CUDA_SUCCESS {
		fmt.Println(cu.GetErrorString(status))
	}
}

func main() {
	// Init Device
	errCheck(cu.Setup(0))

	// input data
	A := []int{100, 100, 100}
	B := []int{10, 10, 10}
	C := []int{0, 0, 0}

	// Malloc DevicePtr
	devA, err := cu.MemAlloc(uint32(unsafe.Sizeof(A)))
	errCheck(err)
	devB, err := cu.MemAlloc(uint32(unsafe.Sizeof(B)))
	errCheck(err)
	devC, err := cu.MemAlloc(uint32(unsafe.Sizeof(C)))
	errCheck(err)

	// transfoer H to D
	errCheck(cu.MemcpyHtoD(devA, &A[0], uint32(unsafe.Sizeof(A))))
	errCheck(cu.MemcpyHtoD(devB, &B[0], uint32(unsafe.Sizeof(B))))

	// load kernel module
	module, err := cu.CreateModuleFromFile("cu/kernel.cu")
	errCheck(err)
	add, err := cu.ModuleGetFunction(module, "add")
	errCheck(err)
	kernelParams := []*cu.CUdeviceptr{&devC, &devA, &devB}

	// launch kernel
	errCheck(cu.LaunchKernel(add, 16, 1, 1, 16, 1, 1, 0, nil, kernelParams, nil))

	// transfoer D to H
	errCheck(cu.MemcpyDtoH(&C[0], devC, uint32(unsafe.Sizeof(C))))

	fmt.Println(B, "+", A, "=", C)
	errCheck(cu.MemFree(devA))
	errCheck(cu.MemFree(devB))
	errCheck(cu.MemFree(devC))
	errCheck(cu.ModuleUnLoad(module))
	errCheck(cu.Teardown(0))
}
