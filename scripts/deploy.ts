/**
 * Deploy SigilRegistry and record the deployment as a machine-checkable fact.
 *
 * The record written here is not documentation. Before it is written, the runtime
 * bytecode at the new address is compared byte for byte against the compiled artifact,
 * and a mismatch aborts the deployment. That check exists because a hand-written
 * deployment record silently drifted once: an edit to a *comment* in the contract changed
 * solc's metadata hash, so the repository source no longer produced the deployed bytecode
 * while every file still claimed it did.
 */

import { execSync } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { network } from "hardhat";

const ARTIFACT = "artifacts/contracts/SigilRegistry.sol/SigilRegistry.json";
const BUILD_INFO_DIR = "artifacts/build-info";

const { ethers, networkName } = await network.create();
const [deployer] = await ethers.getSigners();

const providerNetwork = await ethers.provider.getNetwork();
const chainId = Number(providerNetwork.chainId);
if (networkName === "sepolia" && chainId !== 11155111) {
  throw new Error(`Refusing deployment to unexpected chain ${chainId}`);
}

const artifact = JSON.parse(readFileSync(ARTIFACT, "utf8"));
const expectedRuntime: string =
  typeof artifact.deployedBytecode === "string"
    ? artifact.deployedBytecode
    : artifact.deployedBytecode.object;

console.log(
  `Deploying SigilRegistry to ${networkName} (chain ${chainId}) from ${deployer.address}`
);
const registry = await ethers.deployContract("SigilRegistry");
await registry.waitForDeployment();

const address = await registry.getAddress();
const deploymentTx = registry.deploymentTransaction();
const receipt = await deploymentTx?.wait();

// The deployment is only useful if the deployed code is the code in this repository.
// Comparing here means a mismatch is a failed deploy, not a discovery weeks later.
const onChainRuntime = await ethers.provider.getCode(address);
if (onChainRuntime.toLowerCase() !== expectedRuntime.toLowerCase()) {
  throw new Error(
    `Deployed runtime bytecode does not match ${ARTIFACT}. ` +
      `Recompile from a clean tree and redeploy; do not record this deployment.`
  );
}

const sha256 = (hex: string) =>
  createHash("sha256")
    .update(Buffer.from(hex.replace(/^0x/, ""), "hex"))
    .digest("hex");

const buildInfoName = execSync(
  `ls ${BUILD_INFO_DIR} | grep -v output | head -1`
)
  .toString()
  .trim();
const buildInfo = JSON.parse(
  readFileSync(`${BUILD_INFO_DIR}/${buildInfoName}`, "utf8")
);
const settings = buildInfo.input.settings ?? {};

const record = {
  contract: "SigilRegistry",
  network: networkName,
  chainId,
  address,
  deployer: deployer.address,
  deploymentTx: deploymentTx?.hash ?? null,
  blockNumber: receipt?.blockNumber ?? null,
  gasUsed: receipt ? Number(receipt.gasUsed) : null,
  compiler: buildInfo.solcLongVersion,
  evmVersion: settings.evmVersion ?? "default",
  optimizer: settings.optimizer ?? { enabled: false },
  sourceCommit: execSync("git rev-parse HEAD").toString().trim(),
  sourceDirty:
    execSync("git status --porcelain -- contracts").toString().trim().length >
    0,
  runtimeBytecodeSha256: sha256(onChainRuntime),
  explorer: `https://sepolia.etherscan.io/address/${address}`,
  sourcify: `https://repo.sourcify.dev/${chainId}/${address}/`,
  abiSource: "sigil/chain.py REGISTRY_ABI",
};

mkdirSync("deployments", { recursive: true });
writeFileSync(
  `deployments/${networkName}.json`,
  `${JSON.stringify(record, null, 2)}\n`
);

// The exact input solc consumed, so anyone can reproduce the build or paste it straight
// into a block explorer's standard-JSON verification form.
writeFileSync(
  `deployments/${networkName}-standard-input.json`,
  `${JSON.stringify(buildInfo.input, null, 2)}\n`
);

console.log(`SigilRegistry deployed to: ${address}`);
console.log(
  `Deployment transaction:    ${deploymentTx?.hash ?? "unavailable"}`
);
console.log(`Runtime bytecode verified against ${ARTIFACT}`);
console.log(`Wrote deployments/${networkName}.json`);
console.log(`Set CONTRACT_ADDRESS=${address} in .env`);
