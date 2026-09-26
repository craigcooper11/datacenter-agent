# Datacenter Agent

Diagnoses disk-space problems on your Proxmox containers using Grafana metrics, confirms them with
the Proxmox API, and recommends right-sizing (CPU, memory, disk) from measured usage. It can restart
or resize a container, but only after you click **Approve**.

VMs are listed with their disk size only: Proxmox can't see usage inside a VM unless the QEMU guest
agent is running in it.

Every conversation is traced in MLflow.
