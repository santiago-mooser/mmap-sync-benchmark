pgrep -P 2 | xargs -i taskset -p -c 0 {}
find /sys/devices/virtual/workqueue -name cpumask  -exec sh -c 'echo 1 > {}' ';'
sudo tee "performance" /sys/devices/system/cpu/cpu2/cpufreq/scaling_governor