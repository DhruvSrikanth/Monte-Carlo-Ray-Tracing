# Environment variables
PYTHON = python3
GCC = g++-11

# Checks
VISUALIZE = "true"

# Input to cpp
N_rays = 1e7
N_gridpoints = 1000

# Run sequence
TARGETS = output simulation visualize clean_runtime

# Run the following
target: $(TARGETS)

output:
	mkdir output

# Compile the following
ray_tracing: ./ray_tracing.cpp
	$(GCC) ./ray_tracing.cpp -O3 -ffast-math -march=native -mtune=native -lm

# Execute the following
simulation: ./ray_tracing
	@echo "Simulating...\n"
	./ray_tracing $(N_rays) $(N_gridpoints)
	@echo "\nSimulation Complete!\n"

# Visualize simulation
visualize:
ifeq ($(VISUALIZE), "true")
	@echo "Visualizing...\n"
	$(PYTHON) ./visualize.py
	@echo "\nVisualization Complete!\n"	
endif

# Clean the following
clean_runtime:
	@echo "Cleaning runtime files...\n"
	rm -rf __pycache__
	rm ray_tracing

clean_text:
	@echo "Cleaning text files...\n"
	rm ./output/*.txt

clean_output:
	@echo "Cleaning output files...\n"
	rm -rf ./output