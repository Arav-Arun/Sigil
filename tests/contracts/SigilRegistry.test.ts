import assert from "node:assert/strict";
import { describe, it, beforeEach } from "node:test";
import { network } from "hardhat";

const { ethers } = await network.connect();

const ROOT_A = "0x" + "11".repeat(32);
const ROOT_B = "0x" + "22".repeat(32);
const ZERO = "0x" + "00".repeat(32);
const SCHEMA = 1;

describe("SigilRegistry", () => {
  let registry: any;
  let alice: any;
  let bob: any;

  beforeEach(async () => {
    [alice, bob] = await ethers.getSigners();
    registry = await ethers.deployContract("SigilRegistry");
    await registry.waitForDeployment();
  });

  it("anchors a root and emits the full record", async () => {
    await assert.doesNotReject(registry.anchor(ROOT_A, SCHEMA));

    const [exists, submitter, anchoredAt, schemaVersion] = await registry.verify(ROOT_A);
    assert.equal(exists, true);
    assert.equal(submitter, alice.address);
    assert.equal(schemaVersion, BigInt(SCHEMA));
    assert.ok(anchoredAt > 0n, "anchoredAt must be a real timestamp");
    assert.equal(await registry.totalAnchored(), 1n);
  });

  it("emits Anchored with indexed root and submitter", async () => {
    const tx = await registry.anchor(ROOT_A, SCHEMA);
    const receipt = await tx.wait();
    const log = receipt.logs.find((l: any) => l.fragment?.name === "Anchored");
    assert.ok(log, "Anchored event must be emitted");
    assert.equal(log.args.root, ROOT_A);
    assert.equal(log.args.submitter, alice.address);
    assert.equal(log.args.schemaVersion, BigInt(SCHEMA));
  });

  it("reports an unknown root as absent rather than reverting in verify()", async () => {
    const [exists, submitter, anchoredAt] = await registry.verify(ROOT_B);
    assert.equal(exists, false);
    assert.equal(submitter, ethers.ZeroAddress);
    assert.equal(anchoredAt, 0n);
    assert.equal(await registry.isAnchored(ROOT_B), false);
  });

  it("reverts get() for an unknown root so a zero struct cannot read as success", async () => {
    await assert.rejects(registry.get(ROOT_B), (error: any) => {
      assert.match(error.message, /UnknownRoot/);
      return true;
    });
  });

  it("rejects the zero root", async () => {
    await assert.rejects(registry.anchor(ZERO, SCHEMA), (error: any) => {
      assert.match(error.message, /ZeroRoot/);
      return true;
    });
    assert.equal(await registry.totalAnchored(), 0n);
  });

  it("rejects a duplicate root and preserves the original record", async () => {
    await registry.anchor(ROOT_A, SCHEMA);
    const [, originalSubmitter, originalTime] = await registry.verify(ROOT_A);

    await assert.rejects(registry.connect(bob).anchor(ROOT_A, 2), (error: any) => {
      assert.match(error.message, /AlreadyAnchored/);
      return true;
    });

    const [, submitter, anchoredAt, schemaVersion] = await registry.verify(ROOT_A);
    assert.equal(submitter, originalSubmitter, "submitter must not be overwritten");
    assert.equal(anchoredAt, originalTime, "timestamp must not be overwritten");
    assert.equal(schemaVersion, BigInt(SCHEMA), "schema version must not be overwritten");
  });

  it("keeps roots from different submitters independent", async () => {
    await registry.anchor(ROOT_A, SCHEMA);
    await registry.connect(bob).anchor(ROOT_B, 7);

    const [, submitterA, , schemaA] = await registry.verify(ROOT_A);
    const [, submitterB, , schemaB] = await registry.verify(ROOT_B);

    assert.equal(submitterA, alice.address);
    assert.equal(submitterB, bob.address);
    assert.equal(schemaA, 1n);
    assert.equal(schemaB, 7n);
    assert.equal(await registry.totalAnchored(), 2n);
  });

  it("exposes the anchor through the public mapping identically to verify()", async () => {
    await registry.anchor(ROOT_A, SCHEMA);
    const viaMapping = await registry.anchors(ROOT_A);
    const [, submitter, anchoredAt, schemaVersion] = await registry.verify(ROOT_A);
    assert.equal(viaMapping.submitter, submitter);
    assert.equal(viaMapping.anchoredAt, anchoredAt);
    assert.equal(viaMapping.schemaVersion, schemaVersion);
  });

  it("stays within a sane gas budget for a single anchor", async () => {
    const receipt = await (await registry.anchor(ROOT_A, SCHEMA)).wait();
    assert.ok(
      receipt.gasUsed < 80_000n,
      `anchor() used ${receipt.gasUsed} gas, expected under 80k`,
    );
  });
});
