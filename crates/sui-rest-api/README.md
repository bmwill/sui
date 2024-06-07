# Sui REST API

This is the entry point for the on-node REST API.

## Motivation

Historically Sui Fullnodes have provided a JSON-RPC api for interacting with and querying chain state. Unfortunatley it has a number of design flaws and other scalability concerns. Examples of some of these issues are:

- Lack of consistency between apis (eg paging)
- Some apis require indexed data that requires full chain history to properly create and return accurate responses (eg Object Display standard)
- Too expresive of querying options leading to very expensive and large indexes
- The indexes are presently unprunable meaning they grow unbounded. While some of them may be prunable, if others were pruned it would result in incorrect responses.
- While some data can be pruned from fullnodes (Transactions, objects, etc), if that purned data is requested, by default, fullnodes fall back to a centralized key-value service in order to fetch this pruned data. This is a centralization concern as well as a security risk in the event invalid or incorrect data is served from that key-value service.

All of these concerns, and other usability issues, have lead the team to deprecate JSON-RPC and provide two different APIs to replace it.
Graphql backed by an indexer for more query expresivity and this new REST api for simpler queries backed by a Fullnode.

## Goal

The main goals of this new REST API are as follows:

- Provide a simple and basic api set for interacting with a Sui Fullnode and fetching on-chain data.
- Only support apis which can be backed by prunable data.
    In particular the correctness of a response that can be served (because the data for that response hasn't been pruned yet) should not be contingent on other, previously pruned data.
    As an example, the REST api cannot support the Object Display standard because determining the valid/correct value of an Object's Display format requires full event history.

    Another way to describe this goal would be to say that we require that all indexes be fully derivable when restoring a fullnode from a formal snapshot.
- Support sui-core's testsuite.
- Have support, in select APIs, for the client requesting either JSON or BCS data via the `Accept` header.
