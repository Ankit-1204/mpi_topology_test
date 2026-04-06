## Running Benchmarking for MPI functions : 
- Topologies :
   - Dragonfire
   - Fat Tree
   - Torus

- MPI Functions
   - AlltoAll
   - Reduce
   - Broadcast
   - AllReduce
   - Scatter
   - Point to Point Ring

Run following commands
```
sst++ -I. -fPIC -c ./src/exp_collectives.cc -o exp_collectives.o

sst++ -o exp_collectives exp_collectives.o -Wl,-rpath,/lib   -fPIC

pysstmac -f ./topologies/dragonfly.ini
pysstmac -f ./topologies/fat_tree.ini
pysstmac -f ./topologies/torus.ini

```