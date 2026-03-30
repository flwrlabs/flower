---
myst:
  html_meta:
    description: Deploy Flower's SuperNode Helm chart to install client federated learning components. Default config mirrors official releases for seamless integration.
    property:og:description: Deploy Flower's SuperNode Helm chart to install client federated learning components. Default config mirrors official releases for seamless integration.
---

# Deploy SuperNode using Helm

```{note}
Flower Helm charts are a Flower Enterprise feature. See [Flower Enterprise](https://flower.ai/enterprise) for details.
```

The Flower Framework offers a unified approach to federated learning, analytics, and evaluation,
allowing you to federate any workload, machine learning framework, or programming language.

This Helm chart installs the client-side components of the Flower Framework, specifically
setting up the SuperNode.

The default installation configuration aims to replicate the functionality and setup of the
provided Flower Framework releases.

## Multi Project Setup

To install multiple types of SuperNodes, such as a federation for running PyTorch and another for
TensorFlow, you need to install the Helm Chart multiple times with different names. This allows
each deployment to have its own configurations and dependencies.

For instance, you can install the Chart for the PyTorch setup by adjusting the values.yaml file
as shown below:

```yaml
supernode:
  superlink:
    address: my-superlink.example.com
    port: 9092
  node:
    config:
      partition-id: 0
      num-partitions: 2
  image:
    registry: myregistry.example.com
    repository: flwr/supernode
    tag: 1.20.0-pytorch
```

Install this configuration using the following command:

```sh
$ helm install pytorch . --values values.yaml
```

This will deploy 10 SuperNodes named `pytorch-flower-client-supernode-<random>`.

For a TensorFlow setup, modify the `values.yaml` file as follows:

```yaml
supernode:
  replicas: 3
  superlink:
    address: my-other-superlink.example.com
    port: 9092
  node:
    config:
      partition-id: 1
      num-partitions: 2
  image:
    registry: myregistry.example.com
    repository: flwr/supernode
    tag: 1.20.0-tensorflow
```

Install this configuration using the following command:

```sh
$ helm install tensorflow . --values values.yaml
```

This will deploy 3 SuperNodes named `tensorflow-flower-client-supernode-<random>`.

## Deploy Flower Framework with TLS

By default, the Flower Framework is deployed with TLS enabled. This means `tls.enabled` is
set to `true`.

When using private CAs, the SuperNode must trust the CA certificate in order to connect
securely to the SuperLink.

To provide the CA certificate, set `tls.enabled` to `true` and create a `Secret` of type
`kubernetes.io/tls` named `flower-client-tls`:

```yaml
tls:
  enabled: true
```

If you want to use a different `Secret` name, override the default by setting
`supernode.superlink.certificate.existingSecret`:

```yaml
tls:
  enabled: true

supernode:
  superlink:
    certificate:
      existingSecret: my-custom-tls-secret-name
```

**Important:**

The recommended practice is to mount different `Secret`s for the SuperLink and the
SuperNodes `existingSecret` parameter. Keeping these `Secrets` separate ensures
that if the `Secret` containing the server’s private key and certificate is ever
tampered with, the client will fail to connect rather than trusting a compromised
server.

For further details, refer to the [`cert-manager` documentation](https://cert-manager.io/docs/trust/).

If the SuperLink certificate (of type `kubernetes.io/tls`) is deployed in the same cluster and
namespace as the SuperNode, you can enable `supernode.superlink.certificate.copyFromExistingSecret`.
This instructs the chart to create a new `Secret` containing the CA certificate.
It copies `ca.crt` from the SuperLink `Secret`, or falls back to `tls.crt` if `ca.crt` is not
present.

By default, the copied `Secret` is named `flower-client-tls`. You can customize this name with
`supernode.superlink.certificate.copyFromExistingSecret.secretName`:

```yaml
tls:
  enabled: true

supernode:
  superlink:
    certificate:
      existingSecret: superlink-tls-secret-name
    copyFromExistingSecret:
      enabled: true
      secretName: my-custom-tls-secret-name
```

## Deploy Flower Framework without TLS

You might want to deploy the Flower framework without TLS for testing or internal use. Be
cautious as this exposes your deployment to potential security risks.

```yaml
tls:
  enabled: false
```

## Node Authentication

To enable Node Authentication, you need to specify a private key in either PKCS8 or OpenSSH
(PEM-like) format. This example assumes that the SuperLink is also configured for Node
Authentication and recognizes the `ecdsa-sha2-nistp384 [...]` public key of this SuperNode.

```yaml
global:
  nodeAuth:
    enabled: true
    authSupernodePrivateKey: |+
      -----BEGIN OPENSSH PRIVATE KEY-----
      [...]
      -----END OPENSSH PRIVATE KEY-----
    authSupernodePublicKey: ecdsa-sha2-nistp384 [...]

tls:
  enabled: true

supernode:
  enabled: true
  superlink:
    address: my-superlink.example.com
    port: 9092

superexec:
  enabled: true
  supernode:
    address: my-supernode.example.com
    port: 9094
```

## Isolated Setup

### Isolation All-in-One

To install SuperNode in isolation mode using the "process" configuration, both the SuperExec and
SuperNode need to be enabled. By default, the SuperExec connects to the SuperNode internally
within the cluster, so there is no need to set `supernode.address` and `supernode.port` unless the
connection is external. This setup assumes that both components are running within the same cluster.

```yaml
supernode:
  enabled: true
  isolationMode: process

superexec:
  enabled: true
```

### Isolation Distributed

You can also deploy the SuperNode and SuperExec separately. To do this, you need to deploy the
chart twice: once with `supernode.enabled=true` and once with `superexec.enabled=true`.

```yaml
supernode:
  enabled: true

superexec:
  enabled: true
  supernode:
    address: my-supernode.example.com
    port: 9094
```

## Node Configuration

You can add a node configuration to configure a SuperNode. The YAML datatype is preserved when
passing it in the Python application:

```yaml
supernode:
  node:
    config:
      bool: false
      int: 1
      negative_int: -1
      float: 21.23
      negative_float: -1.34
      string: value 1
      int-as-string: "1"
```

## Parameters

### Helm parameters

| Name               | Description                                      | Value           |
| ------------------ | ------------------------------------------------ | --------------- |
| `nameOverride`     | Replaces the name of the chart in the Chart.yaml | `flower-client` |
| `fullnameOverride` | Completely replaces the generated name.          | `""`            |

### Global parameters

| Name                                                 | Description                                                       | Value            |
| ---------------------------------------------------- | ----------------------------------------------------------------- | ---------------- |
| `global.annotations`                                 | Default Annotations                                               | `{}`             |
| `global.labels`                                      | Default Labels                                                    | `{}`             |
| `global.podLabels`                                   | Default PodLabels                                                 | `{}`             |
| `global.nodeSelector`                                | Default node selector for all components                          | `{}`             |
| `global.tolerations`                                 | Default tolerations for all components                            | `[]`             |
| `global.affinity.podAntiAffinity`                    | Default affinity preset for all components                        | `soft`           |
| `global.affinity.podAntiAffinity`                    | Default pod anti-affinity rules. Either: `none`, `soft` or `hard` | `soft`           |
| `global.affinity.nodeAffinity.type`                  | Default node affinity rules. Either: `none`, `soft` or `hard`     | `hard`           |
| `global.affinity.nodeAffinity.matchExpressions`      | Default match expressions for node affinity                       | `[]`             |
| `global.nodeAuth.enabled`                            | Enables or Disables Node-Authentication SuperLink \<-> SuperNode  | `false`          |
| `global.nodeAuth.authSupernodePrivateKey`            | Specifies the ecdsa-sha2-nistp384 private key                     | `""`             |
| `global.securityContext.runAsUser`                   | Set Security Context runAsUser                                    | `49999`          |
| `global.securityContext.runAsGroup`                  | Set Security Context runAsGroup                                   | `49999`          |
| `global.securityContext.fsGroup`                     | Set Security Context fsGroup                                      | `49999`          |
| `global.podSecurityContext.runAsNonRoot`             | Set Security Context runAsNonRoot                                 | `true`           |
| `global.podSecurityContext.readOnlyRootFilesystem`   | Set Security Context readOnlyRootFilesystem                       | `true`           |
| `global.podSecurityContext.allowPrivilegeEscalation` | Set Security Context allowPrivilegeEscalation                     | `false`          |
| `global.podSecurityContext.seccompProfile.type`      | Set Security Context seccompProfile                               | `RuntimeDefault` |
| `global.podSecurityContext.capabilities.drop`        | Set Security Context capabilities                                 | `["ALL"]`        |
| `global.env`                                         | Default environment variables                                     | `[]`             |
| `global.image.pullPolicy`                            | Default image pullPolicy                                          | `IfNotPresent`   |

### TLS Configuration

| Name          | Description                                        | Value  |
| ------------- | -------------------------------------------------- | ------ |
| `tls.enabled` | Enable TLS configuration for the Flower Framework. | `true` |

### Component SuperNode

| Name                                                                | Description                                                                                                             | Value                      |
| ------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- | -------------------------- |
| `supernode.name`                                                    | Name of the SuperNode                                                                                                   | `supernode`                |
| `supernode.enabled`                                                 | Enable or disable SuperNode                                                                                             | `true`                     |
| `supernode.resources`                                               | Set container requests and limits for different resources like CPU or memory (essential for production workloads)       | `{}`                       |
| `supernode.node.config`                                             |                                                                                                                         | `{}`                       |
| `supernode.isolationMode`                                           | The isolation mode of the SuperNode                                                                                     | `subprocess`               |
| `supernode.resources`                                               | Set container requests and limits for different resources like CPU or memory (essential for production workloads)       | `{}`                       |
| `supernode.superlink.address`                                       | Address of the SuperLink the SuperNodes should connect to                                                               | `my-superlink.example.com` |
| `supernode.superlink.port`                                          | Port of the SuperLink the SuperNodes should connect to                                                                  | `9092`                     |
| `supernode.superlink.certificate.existingSecret`                    |                                                                                                                         | `""`                       |
| `supernode.superlink.certificate.copyFromExistingSecret.enabled`    |                                                                                                                         | `false`                    |
| `supernode.superlink.certificate.copyFromExistingSecret.secretName` |                                                                                                                         | `""`                       |
| `supernode.volumes`                                                 | Specify a list of volumes for the SuperNode pod(s)                                                                      | `[]`                       |
| `supernode.volumeMounts`                                            | Allows to specify additional VolumeMounts                                                                               | `[]`                       |
| `supernode.automountServiceAccountToken`                            | Automount SA-Token into the pod.                                                                                        | `true`                     |
| `supernode.serviceAccount.enabled`                                  | Enable a service account for this component                                                                             | `true`                     |
| `supernode.serviceAccount.annotations`                              | Annotations applied to enabled service account                                                                          | `{}`                       |
| `supernode.serviceAccount.labels`                                   | Labels applied to enabled service account                                                                               | `{}`                       |
| `supernode.serviceAccount.automountServiceAccountToken`             | Automount SA-Token                                                                                                      | `true`                     |
| `supernode.service.type`                                            | Valid are ClusterIP, NodePort or Loadbalancer                                                                           | `ClusterIP`                |
| `supernode.service.servicePortClientAppIoName`                      | Prefix of the SuperNode ClientAppIo API port                                                                            | `clientappio`              |
| `supernode.service.servicePortClientAppIo`                          | Port to expose for the SuperNode ClientAppIo API                                                                        | `9094`                     |
| `supernode.service.nodePortClientAppIo`                             | Node port for SuperNode ClientAppIo API                                                                                 | `""`                       |
| `supernode.containerPorts.clientAppIo`                              | Container port for SuperNode ClientAppIo API                                                                            | `9094`                     |
| `supernode.containerPorts.health`                                   | Container port for SuperNode Health API                                                                                 | `8081`                     |
| `supernode.podSecurityContext`                                      |                                                                                                                         | `{}`                       |
| `supernode.replicas`                                                | The number of SuperNode pods to run                                                                                     | `1`                        |
| `supernode.labels`                                                  | Extra labels for SuperNode pods                                                                                         | `{}`                       |
| `supernode.extraArgs`                                               | Add extra arguments to the default arguments for the SuperNode                                                          | `[]`                       |
| `supernode.nodeSelector`                                            | Node labels for SuperNode pods which merges with global.nodeSelector                                                    | `{}`                       |
| `supernode.tolerations`                                             | Node tolerations for SuperNode pods which merges with global.tolerations                                                | `[]`                       |
| `supernode.updateStrategy.type`                                     | SuperNode deployment strategy type                                                                                      | `RollingUpdate`            |
| `supernode.updateStrategy.rollingUpdate`                            | SuperNode deployment rolling update configuration parameters                                                            | `{}`                       |
| `supernode.affinity`                                                | Node affinity for SuperNode pods which merges with global.affinity                                                      | `{}`                       |
| `supernode.env`                                                     | Array with extra environment variables to add to SuperNode nodes which merges with global.env                           | `[]`                       |
| `supernode.livenessProbe.enabled`                                   | Enable livenessProbe on SuperNode containers                                                                            | `true`                     |
| `supernode.livenessProbe.initialDelaySeconds`                       | Initial delay seconds for livenessProbe                                                                                 | `0`                        |
| `supernode.livenessProbe.periodSeconds`                             | Period seconds for livenessProbe                                                                                        | `10`                       |
| `supernode.livenessProbe.timeoutSeconds`                            | Timeout seconds for livenessProbe                                                                                       | `1`                        |
| `supernode.livenessProbe.failureThreshold`                          | Failure threshold for livenessProbe                                                                                     | `3`                        |
| `supernode.livenessProbe.successThreshold`                          | Success threshold for livenessProbe                                                                                     | `1`                        |
| `supernode.readinessProbe.enabled`                                  | Enable readinessProbe on SuperNode containers                                                                           | `true`                     |
| `supernode.readinessProbe.initialDelaySeconds`                      | Initial delay seconds for readinessProbe                                                                                | `0`                        |
| `supernode.readinessProbe.periodSeconds`                            | Period seconds for readinessProbe                                                                                       | `10`                       |
| `supernode.readinessProbe.timeoutSeconds`                           | Timeout seconds for readinessProbe                                                                                      | `1`                        |
| `supernode.readinessProbe.failureThreshold`                         | Failure threshold for readinessProbe                                                                                    | `3`                        |
| `supernode.readinessProbe.successThreshold`                         | Success threshold for readinessProbe                                                                                    | `1`                        |
| `supernode.lifecycle`                                               | SuperNode container(s) to automate configuration before or after startup                                                | `{}`                       |
| `supernode.annotations`                                             | Additional custom annotations for SuperNode                                                                             | `{}`                       |
| `supernode.selectorLabels`                                          | Extra selectorLabels for SuperNode pods                                                                                 | `{}`                       |
| `supernode.podAnnotations`                                          | Annotations for SuperNode pods                                                                                          | `{}`                       |
| `supernode.podLabels`                                               | Extra podLabels for SuperNode pods                                                                                      | `{}`                       |
| `supernode.imagePullSecrets`                                        | SuperNode image pull secrets which overrides global.imagePullSecrets                                                    | `[]`                       |
| `supernode.image.registry`                                          | SuperNode image registry                                                                                                | `registry.hub.docker.com`  |
| `supernode.image.repository`                                        | SuperNode image repository                                                                                              | `flwr/supernode-ee`        |
| `supernode.image.tag`                                               | Image tag of SuperNode                                                                                                  | `1.26.1-ubuntu`            |
| `supernode.image.digest`                                            | Image digest of SuperNode                                                                                               | `""`                       |
| `supernode.image.pullPolicy`                                        | Components image pullPolicy                                                                                             | `IfNotPresent`             |
| `supernode.networkPolicy.enabled`                                   | Specifies whether a NetworkPolicy should be created                                                                     | `true`                     |
| `supernode.networkPolicy.allowExternal`                             | Allow external ingress traffic                                                                                          | `true`                     |
| `supernode.networkPolicy.allowExternalEgress`                       | Allow unrestricted egress traffic                                                                                       | `true`                     |
| `supernode.networkPolicy.extraIngress`                              | Add extra ingress rules to the NetworkPolicy                                                                            | `[]`                       |
| `supernode.networkPolicy.extraEgress`                               | Add extra ingress rules to the NetworkPolicy (ignored if allowExternalEgress=true)                                      | `[]`                       |
| `supernode.networkPolicy.ingressPodMatchLabels`                     | Labels to match to allow traffic from other pods. Ignored if `supernode.networkPolicy.allowExternal` is true.           | `{}`                       |
| `supernode.networkPolicy.ingressNSMatchLabels`                      | Labels to match to allow traffic from other namespaces. Ignored if `supernode.networkPolicy.allowExternal` is true.     | `{}`                       |
| `supernode.networkPolicy.ingressNSPodMatchLabels`                   | Pod labels to match to allow traffic from other namespaces. Ignored if `supernode.networkPolicy.allowExternal` is true. | `{}`                       |

### Component SuperExec

| Name                                                    | Description                                                                                                       | Value                     |
| ------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- | ------------------------- |
| `superexec.name`                                        | Name of the SuperExec                                                                                             | `superexec-clientapp`     |
| `superexec.enabled`                                     | Enable or disable SuperExec component                                                                             | `false`                   |
| `superexec.pluginType`                                  | The type of plugin to use.                                                                                        | `clientapp`               |
| `superexec.resources`                                   | Set container requests and limits for different resources like CPU or memory (essential for production workloads) | `{}`                      |
| `superexec.supernode`                                   | Address of the supernode the SuperExec should connect to                                                          | `{}`                      |
| `superexec.volumes`                                     | Specify a list of volumes for the SuperExec pod(s)                                                                | `[]`                      |
| `superexec.volumeMounts`                                | Allows to specify additional VolumeMounts                                                                         | `[]`                      |
| `superexec.automountServiceAccountToken`                | Automount SA-Token into the pod.                                                                                  | `true`                    |
| `superexec.serviceAccount.enabled`                      | Enable a service account for this component                                                                       | `true`                    |
| `superexec.serviceAccount.annotations`                  | Annotations applied to enabled service account                                                                    | `{}`                      |
| `superexec.serviceAccount.labels`                       | Labels applied to enabled service account                                                                         | `{}`                      |
| `superexec.serviceAccount.automountServiceAccountToken` | Automount SA-Token                                                                                                | `true`                    |
| `superexec.containerPorts.health`                       | Container port for SuperExec Health API                                                                           | `8081`                    |
| `superexec.podSecurityContext`                          | Security settings for the SuperExec Pod                                                                           | `{}`                      |
| `superexec.livenessProbe.enabled`                       | Enable livenessProbe on SuperExec containers                                                                      | `true`                    |
| `superexec.livenessProbe.initialDelaySeconds`           | Initial delay seconds for livenessProbe                                                                           | `0`                       |
| `superexec.livenessProbe.periodSeconds`                 | Period seconds for livenessProbe                                                                                  | `10`                      |
| `superexec.livenessProbe.timeoutSeconds`                | Timeout seconds for livenessProbe                                                                                 | `1`                       |
| `superexec.livenessProbe.failureThreshold`              | Failure threshold for livenessProbe                                                                               | `3`                       |
| `superexec.livenessProbe.successThreshold`              | Success threshold for livenessProbe                                                                               | `1`                       |
| `superexec.readinessProbe.enabled`                      | Enable readinessProbe on SuperExec containers                                                                     | `true`                    |
| `superexec.readinessProbe.initialDelaySeconds`          | Initial delay seconds for readinessProbe                                                                          | `0`                       |
| `superexec.readinessProbe.periodSeconds`                | Period seconds for readinessProbe                                                                                 | `10`                      |
| `superexec.readinessProbe.timeoutSeconds`               | Timeout seconds for readinessProbe                                                                                | `1`                       |
| `superexec.readinessProbe.failureThreshold`             | Failure threshold for readinessProbe                                                                              | `3`                       |
| `superexec.readinessProbe.successThreshold`             | Success threshold for readinessProbe                                                                              | `1`                       |
| `superexec.replicas`                                    | The number of SuperExec containers to run                                                                         | `1`                       |
| `superexec.labels`                                      | Extra labels for SuperExec pods                                                                                   | `{}`                      |
| `superexec.extraArgs`                                   | Add extra arguments to the default arguments for the SuperExec                                                    | `[]`                      |
| `superexec.nodeSelector`                                | Node labels for SuperExec pods which merges with global.nodeSelector                                              | `{}`                      |
| `superexec.tolerations`                                 | Node tolerations for SuperExec pods which merges with global.tolerations                                          | `[]`                      |
| `superexec.updateStrategy.type`                         | SuperExec deployment strategy type                                                                                | `RollingUpdate`           |
| `superexec.updateStrategy.rollingUpdate`                | SuperExec deployment rolling update configuration parameters                                                      | `{}`                      |
| `superexec.affinity`                                    | Node affinity for SuperExec pods which merges with global.affinity                                                | `{}`                      |
| `superexec.env`                                         | Array with extra environment variables to add to SuperExec nodes which merges with global.env                     | `[]`                      |
| `superexec.lifecycle`                                   | SuperExec container(s) to automate configuration before or after startup                                          | `{}`                      |
| `superexec.annotations`                                 | Additional custom annotations for SuperExec                                                                       | `{}`                      |
| `superexec.selectorLabels`                              | Extra selectorLabels for SuperExec pods                                                                           | `{}`                      |
| `superexec.podAnnotations`                              | Annotations for SuperExec pods                                                                                    | `{}`                      |
| `superexec.podLabels`                                   | Extra podLabels for SuperExec pods                                                                                | `{}`                      |
| `superexec.imagePullSecrets`                            | SuperExec image pull secrets which overrides global.imagePullSecrets                                              | `[]`                      |
| `superexec.image.registry`                              | SuperExec image registry                                                                                          | `registry.hub.docker.com` |
| `superexec.image.repository`                            | SuperExec image repository                                                                                        | `flwr/superexec-ee`       |
| `superexec.image.tag`                                   | Image tag of SuperExec                                                                                            | `1.26.1-ubuntu`           |
| `superexec.image.digest`                                | Image digest of SuperExec                                                                                         | `""`                      |
| `superexec.image.pullPolicy`                            | Components image pullPolicy                                                                                       | `IfNotPresent`            |
| `superexec.networkPolicy.enabled`                       | Specifies whether a NetworkPolicy should be created                                                               | `true`                    |
| `superexec.networkPolicy.allowExternalEgress`           | Allow unrestricted egress traffic                                                                                 | `true`                    |
| `superexec.networkPolicy.extraEgress`                   | Add extra ingress rules to the NetworkPolicy (ignored if allowExternalEgress=true)                                | `[]`                      |
