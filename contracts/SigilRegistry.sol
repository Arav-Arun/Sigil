// SPDX-License-Identifier: MIT
pragma solidity 0.8.34;

/// @title SigilRegistry
/// @notice Tamper-evident anchoring for Sigil evidence bundles.
/// @dev The contract is deliberately minimal. All of the interesting work, canonical
///      JSON, domain-separated Merkle leaves, per-field inclusion proofs, happens off
///      chain and is verifiable by anyone holding the bundle. What must be on chain is
///      only the part that needs an immutable, independently timestamped witness: the
///      32-byte root, who submitted it, and when.
///
///      Nothing personal is ever stored here. No images, no embeddings, no names, no
///      post text, only a hash. That is a privacy property, not an optimisation:
///      biometric data must not be written to an immutable public ledger.
contract SigilRegistry {
    /// @param submitter The address that anchored this root.
    /// @param anchoredAt Block timestamp of the anchoring transaction.
    /// @param schemaVersion Evidence-manifest schema the root was built under.
    struct Anchor {
        address submitter;
        uint64 anchoredAt;
        uint16 schemaVersion;
    }

    /// @notice Anchored evidence roots, keyed by the Merkle root itself.
    mapping(bytes32 root => Anchor anchor) public anchors;

    /// @notice Number of roots anchored, exposed for cheap sanity checks.
    uint256 public totalAnchored;

    event Anchored(
        bytes32 indexed root,
        address indexed submitter,
        uint64 anchoredAt,
        uint16 schemaVersion
    );

    /// @dev The zero root is what an uninitialised or failed off-chain build produces.
    ///      Accepting it would let a broken run look successful.
    error ZeroRoot();

    /// @dev Re-anchoring would overwrite the original timestamp and submitter, which is
    ///      precisely the record the proof depends on.
    error AlreadyAnchored(bytes32 root, address submitter, uint64 anchoredAt);

    error UnknownRoot(bytes32 root);

    /// @notice Anchor an evidence root.
    /// @param root The Merkle root of a canonical evidence manifest.
    /// @param schemaVersion The manifest schema version used to build `root`.
    function anchor(bytes32 root, uint16 schemaVersion) external {
        if (root == bytes32(0)) revert ZeroRoot();

        Anchor memory existing = anchors[root];
        if (existing.anchoredAt != 0) {
            revert AlreadyAnchored(root, existing.submitter, existing.anchoredAt);
        }

        uint64 timestamp = uint64(block.timestamp);
        anchors[root] = Anchor({
            submitter: msg.sender,
            anchoredAt: timestamp,
            schemaVersion: schemaVersion
        });
        unchecked {
            ++totalAnchored;
        }

        emit Anchored(root, msg.sender, timestamp, schemaVersion);
    }

    /// @notice Read an anchor, reverting when the root was never anchored.
    /// @dev Verifiers that want a boolean should call `isAnchored` instead; this
    ///      variant exists so a mistaken "verified" cannot come from a zero struct.
    function get(bytes32 root) external view returns (Anchor memory) {
        Anchor memory record = anchors[root];
        if (record.anchoredAt == 0) revert UnknownRoot(root);
        return record;
    }

    /// @notice Non-reverting existence check with the full record.
    function verify(bytes32 root)
        external
        view
        returns (bool exists, address submitter, uint64 anchoredAt, uint16 schemaVersion)
    {
        Anchor memory record = anchors[root];
        exists = record.anchoredAt != 0;
        return (exists, record.submitter, record.anchoredAt, record.schemaVersion);
    }

    /// @notice Cheapest possible existence check.
    function isAnchored(bytes32 root) external view returns (bool) {
        return anchors[root].anchoredAt != 0;
    }
}
