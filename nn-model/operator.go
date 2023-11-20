package operator

import (
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"math/rand"
	"os"
	"os/signal"
	"runtime/pprof"
	"syscall"
	"path/filepath"
	"encoding/binary"
	"io"

	"net/http"
	"net/http/pprof"

	"github.com/pkg/errors"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"

	"time"

	"gopkg.in/cheggaaa/pb.v1"

	"github.com/0xpanicError/EigenLayer-Federated-Learning/nn-model/mnist"
	"github.com/0xpanicError/EigenLayer-Federated-Learning/aggregator"
)

// mnist dataset is located in the data folder in root of project
const loc = "../data"

var dt tensor.Dtype

func trainData(baseParams *parameters) (*parameters, err error) {
	flag.Parse()
	parseDtype()
	rand.Seed(1337)

	// intercept Ctrl+C
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	doneChan := make(chan bool, 1)

	var inputs, targets tensor.Tensor
	var err error

	go func() {
		log.Println(http.ListenAndServe("localhost:6060", nil))
	}()

	trainOn := *dataset
	if inputs, targets, err = mnist.Load(trainOn, loc, dt); err != nil {
		log.Fatal(err)
	}

	// the data is in (numExamples, 784).
	// In order to use a convnet, we need to massage the data
	// into this format (batchsize, numberOfChannels, height, width).
	//
	// This translates into (numExamples, 1, 28, 28).
	//
	// This is because the convolution operators actually understand height and width.
	//
	// The 1 indicates that there is only one channel (MNIST data is black and white).
	numExamples := inputs.Shape()[0]
	bs := *batchsize

	// train given set of input parameters
	g := baseParams.g
	x := baseParams.x
	y := baseParams.y
	m := baseParams.m

	if err = m.fwd(x); err != nil {
		log.Fatalf("%+v", err)
	}
	losses := gorgonia.Must(gorgonia.HadamardProd(m.out, y))
	cost := gorgonia.Must(gorgonia.Mean(losses))
	cost = gorgonia.Must(gorgonia.Neg(cost))

	// we wanna track costs
	var costVal gorgonia.Value
	gorgonia.Read(cost, &costVal)

	prog, locMap, _ := gorgonia.Compile(g)
	log.Printf("%v", prog)

	vm := gorgonia.NewTapeMachine(g, gorgonia.WithPrecompiled(prog, locMap), gorgonia.BindDualValues(m.learnables()...))
	solver := gorgonia.NewRMSPropSolver(gorgonia.WithBatchSize(float64(bs)))
	defer vm.Close()
	// pprof
	// handlePprof(sigChan, doneChan)

	var profiling bool
	if *cpuprofile != "" {
		f, err := os.Create(*cpuprofile)
		if err != nil {
			log.Fatal(err)
		}
		profiling = true
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}
	go cleanup(sigChan, doneChan, profiling)

	batches := numExamples / bs
	log.Printf("Batches %d", batches)
	bar := pb.New(batches)
	bar.SetRefreshRate(time.Second)
	bar.SetMaxWidth(80)

	for i := 0; i < *epochs; i++ {
		bar.Prefix(fmt.Sprintf("Epoch %d", i))
		bar.Set(0)
		bar.Start()
		for b := 0; b < batches; b++ {
			start := b * bs
			end := start + bs
			if start >= numExamples {
				break
			}
			if end > numExamples {
				end = numExamples
			}

			var xVal, yVal tensor.Tensor
			if xVal, err = inputs.Slice(sli{start, end}); err != nil {
				log.Fatal("Unable to slice x")
			}

			if yVal, err = targets.Slice(sli{start, end}); err != nil {
				log.Fatal("Unable to slice y")
			}
			if err = xVal.(*tensor.Dense).Reshape(bs, 1, 28, 28); err != nil {
				log.Fatalf("Unable to reshape %v", err)
			}

			gorgonia.Let(x, xVal)
			gorgonia.Let(y, yVal)
			if err = vm.RunAll(); err != nil {
				log.Fatalf("Failed at epoch  %d: %v", i, err)
			}
			solver.Step(gorgonia.NodesToValueGrads(m.learnables()))
			vm.Reset()
			bar.Increment()
		}
		log.Printf("Epoch %d | cost %v", i, costVal)
	}

	// return the trained parameters
	return baseParams, nil
}

