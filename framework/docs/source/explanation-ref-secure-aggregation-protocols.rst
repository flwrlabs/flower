:og:description: In this Flower explainer, learn what secure aggregation methods are, how they can help in federated learning, and the basics of how the two most common methods work.
.. meta::
    :description: In this Flower explainer, learn what secure aggregation methods are, how they can help in federated learning, and the basics of how the two most common methods work.

####################
 Secure Aggregation
####################

In federated learning, Secure Aggregation methods enable the aggregation server to
compute the aggregated model without ever seeing any individual client’s update.
Individual updates may leak sensitive information about a client’s local data, while
aggregated updates are typically less revealing and harder to link back to any single
contributor. Federations concerned about a malicious or compromised aggregation server
should consider using secure aggregation methods, particularly with small client
datasets, as the risk of information leakage is higher in such cases.

************************************************
 Two Methods: Software-based and Hardware-based
************************************************

Let's discuss two secure aggregation methods, one already supported in Flower and
another that is in development. The first is the cryptographic (software) approach
invented by Google [1]: clients add random masks to their updates and coordinate those
masks so they cancel out only when the server sums across clients. This provides strong
protection against an honest-but-curious server, but adds protocol complexity, extra
communication, and only works for sum-based aggregation methods such as FedAvg.
Furthermore, the honest-but-curious security model may not be strong enough, and
extending this method to a "dishonest server" threat model requires additional algorithm
extensions, including a trusted third-party. See the paper at `Practical Secure
Aggregation for Federated Learning on User-Held Data
<https://arxiv.org/abs/1611.04482>`_.

The second is to run aggregation inside a `confidential VM
<https://en.wikipedia.org/wiki/Confidential_computing>`_ (CVM), a HW-based security
solution. Here, clients can send updates as normal because the aggregation code runs in
an isolated environment designed to stay confidential even from a malicious hypervisor.
This can support more flexible aggregation logic and simpler clients, but shifts trust
to the confidential computing stack (hardware, firmware, and a so-called "attestation"
service) and introduces a different class of operational and side-channel risks.
Furthermore, not all servers/cloud service providers support CVMs.

For details on using Flower's current secure aggregation implementation, see `Flower
SecureAggPlusWorkflow documentation
<https://flower.ai/docs/framework/ref-api/flwr.server.workflow.SecAggPlusWorkflow.html>`_.

*************************************************************
 How the Software-based Method Works: A Three-Client Example
*************************************************************

To illustrate how the software-based cryptographic method works, let's consider a simple
example with three clients (A, B, and C) and a server. Each client has a two-dimensional
update vector that they want to send to the server for aggregation. The goal is for the
server to compute the sum of these updates without learning any individual client's
update.

- Clients A, B and C have the following updates:

.. math::

    u_A = [2,\; 5]

    u_B = [4,\; 1]

    u_C = [3,\; 2]

- So the correct aggregate sum is:

.. math::

    u_A + u_B + u_C = [9,\; 8]

Here's the gist of how the server can compute this without ever seeing the client's
updates.

Clients coordinate pairwise random noise
========================================

First, each pair of clients agrees on a shared random noise vector (see `Diffie–Hellman
key exchange <https://en.wikipedia.org/wiki/Diffie%E2%80%93Hellman_key_exchange>`_).
Each noise vector is associated with an ordered pair of the two clients that share it
(client order determined beforehand):

.. math::

    \text{noise}_{AB} = [7,\; 9]

    \text{noise}_{AC} = [8,\; 6]

    \text{noise}_{BC} = [5,\; 4]

Each client builds a mask from the shared noise
===============================================

Then, each client constructs a mask by adding or subtracting each of its shared noise
vectors. The clients know whether to add or subtract a given noise vector based on if
they are the first or second client in the ordered pair:

.. math::

    \text{mask}_A
    = +\text{noise}_{AB} + \text{noise}_{AC}
    = [7,\; 9] + [8,\; 6]
    = [15,\; 15]

    \text{mask}_B
    = -\text{noise}_{AB} + \text{noise}_{BC}
    = [-7,\; -9] + [5,\; 4]
    = [-2,\; -5]

    \text{mask}_C
    = -\text{noise}_{AC} - \text{noise}_{BC}
    = [-8,\; -6] + [-5,\; -4]
    = [-13,\; -10]

Clients send masked updates to the server
=========================================

Each client sends its masked update by adding its real vector with its mask:

.. math::

    \text{sent}_A = u_A + \text{mask}_A
    = [2,\; 5] + [15,\; 15]
    = [17,\; 20]

    \text{sent}_B = u_B + \text{mask}_B
    = [4,\; 1] + [-2,\; -5]
    = [2,\; -4]

    \text{sent}_C = u_C + \text{mask}_C
    = [3,\; 2] + [-13,\; -10]
    = [-10,\; -8]

The server aggregates masked updates
====================================

The server sums what it receives:

.. math::

    \text{sent}_A + \text{sent}_B + \text{sent}_C
    = [17,\; 20] + [2,\; -4] + [-10,\; -8]
    = [9,\; 8]

This equals the true aggregate!

.. math::

    u_A + u_B + u_C = [9,\; 8]

Why the noise cancels
=====================

If you sum the masks, every pairwise noise term appears once with a plus and once with a
minus:

.. math::

    0 = \text{mask}_A + \text{mask}_B + \text{mask}_C
    =
    (+\text{noise}_{AB} + \text{noise}_{AC})
    + (-\text{noise}_{AB} + \text{noise}_{BC})
    + (-\text{noise}_{AC} - \text{noise}_{BC})

So the server learns only the aggregate, not any individual client update!

However, this doesn't make a full solution. There are quite a few open questions
remaining, such as:

What if a client drops out?
===========================

For drop-outs, a second, more complex secret-sharing scheme is used to enable the
remaining clients to send a vector of the missing noise to the server. See the paper [1]
or the `Flower SecureAggPlusWorkflow documentation
<https://flower.ai/docs/framework/ref-api/flwr.server.workflow.SecAggPlusWorkflow.html>`_
for the details.

So then what if the server lies about a dropout?
================================================

This is where a "dishonest" server can break the security. To protect against this, a
trusted third-party is required to track exactly which clients have sent updates each
round (e.g. a trusted ledger).

And do the clients have to communicate directly?
================================================

Not in practice. Instead, the clients communicate via encrypted messages routed through
the server. Each client posts a public key, which allows others to encrypt messages only
it can decrypt. These encrypted messages are sent to the server and pulled by the
appropriate client (only the intended receiving client will have the private key needed
to decrypt).

Don't the pairwise vectors scale poorly with many clients?
==========================================================

In practice, clients only exchange noise vectors with a small number of "neighbors" in a
communication graph, which can be designed to balance security and communication
overhead.

*********************************************************
 Upcoming Hardware-based Method: Confidential VMs (CVMs)
*********************************************************

We are currently working on a confidential VM (CVM) based secure aggregation method.
CVMs keep running code confidential during execution, even from a malicious hypervisor,
including encrypting code and data in DRAM. They also work with "attestation" services
to allow clients to verify that the correct code is running in the CVM before sending
their updates. This allows clients to send unmasked updates, since the aggregation code
is protected by the CVM running the expected code. Stay tuned for CVM-based secure
aggregation in Flower!

**References:**

[1] Keith Bonawitz, Vladimir Ivanov, Ben Kreuter, Antonio Marcedone, H. Brendan McMahan,
Sarvar Patel, Daniel Ramage, Aaron Segal, and Karn Seth. *Practical Secure Aggregation
for Privacy-Preserving Machine Learning*. Proceedings of the 2017 ACM SIGSAC Conference
on Computer and Communications Security (CCS ’17), 2017.
