// Copyright 2024 Flower Labs GmbH. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================
import { isEqual } from 'lodash';
import { PathLike } from 'fs';
import { RecordSet } from './recordset';
import { Fab, Run, UserConfig, Context } from './typing';
import { getFusedConfig, getFusedConfigFromDir, getFusedConfigFromFab } from './config';

interface RunInfo {
  context: Context;
  initialRunConfig: UserConfig;
}

export class NodeState {
  nodeId: bigint;
  nodeConfig: UserConfig;
  runInfos: { [key: number]: RunInfo };

  constructor(nodeId: bigint, nodeConfig: UserConfig) {
    this.nodeId = nodeId;
    this.nodeConfig = nodeConfig;
    this.runInfos = {};
  }

  // Register a new run context for this node
  registerContext(
    runId: number,
    run: Run | null = null,
    flwrPath: PathLike | null = null,
    appDir: string | null = null,
    fab: Fab | null = null,
  ): void {
    if (!(runId in this.runInfos)) {
      let initialRunConfig: UserConfig = {};

      if (appDir) {
        const appPath = appDir;  // appPath is a string instead of Pathlike in this case

        if (/* Check if appPath is a directory - this needs specific NodeJS code */ true) {
          const overrideConfig = run?.overrideConfig || {};
          // initialRunConfig = getFusedConfigFromDir(appPath, overrideConfig);
        } else {
          throw new Error('The specified `appDir` must be a directory.');
        }
      } else {
        if (run) {
          if (fab) {
            // Load config from FAB and fuse
            // initialRunConfig = getFusedConfigFromFab(fab.content, run);
          } else {
            // Load config from installed FAB and fuse
            // initialRunConfig = getFusedConfig(run, flwrPath);
          }
        } else {
          initialRunConfig = {};
        }
      }

      this.runInfos[runId] = {
        initialRunConfig,
        context: {
          nodeId: this.nodeId,
          nodeConfig: this.nodeConfig,
          state: new RecordSet(),
          runConfig: { ...initialRunConfig },
        } as Context,
      };
    }
  }

  // Retrieve the context given a runId
  retrieveContext(runId: number): Context {
    if (runId in this.runInfos) {
      return this.runInfos[runId].context;
    }

    throw new Error(
      `Context for runId=${runId} doesn't exist. A run context must be registered before it can be retrieved or updated by a client.`,
    );
  }

  // Update run context
  updateContext(runId: number, context: Context): void {
    if (!isEqual(context.runConfig, this.runInfos[runId].initialRunConfig)) {
      throw new Error(
        `The run_config field of the Context object cannot be modified (runId: ${runId}).`,
      );
    }
    this.runInfos[runId].context = context;
  }
}
