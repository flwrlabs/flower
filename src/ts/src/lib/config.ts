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
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import * as toml from '@iarna/toml';
import { APP_DIR, FAB_CONFIG_FILE, FLWR_HOME } from './constants';
import { Run, UserConfig, UserConfigValue } from './typing';

import { PathLike } from 'fs';
import { ZipFile, open } from 'yauzl';
import * as zlib from 'zlib';

type AnyDict = Record<string, any>;

export function getFabConfig(fabFile: PathLike | Buffer): AnyDict {
  /**
   * Extract the config from a FAB file or path.
   * @param fabFile The Flower App Bundle file to validate and extract the metadata from.
   *                It can either be a path to the file or the file itself as bytes.
   */
  // let fabFileArchive: PathLike | Buffer;
  // if (Buffer.isBuffer(fabFile)) {
  //   fabFileArchive = fabFile;
  // } else if (typeof fabFile === 'string') {
  //   fabFileArchive = path.resolve(fabFile);
  // } else {
  //   throw new Error('fabFile must be either a Path or Buffer');
  // }

  // // Unzip the FAB file to read pyproject.toml
  // const zip = open(fabFileArchive.toString());
  // const zipFile = new ZipFile(zip);
  // let tomlContent = '';

  // // Read pyproject.toml from the archive
  // zipFile.on('entry', (entry) => {
  //   if (entry.fileName === 'pyproject.toml') {
  //     const readStream = zipFile.openReadStream(entry);
  //     if (readStream) {
  //       readStream.pipe(zlib.createGunzip()).on('data', (data) => {
  //         tomlContent += data.toString('utf-8');
  //       });
  //     }
  //   }
  // });

  // // Load TOML content
  // const config = loadFromString(tomlContent);
  // if (!config) {
  //   throw new Error('Invalid TOML content in pyproject.toml');
  // }

  // const [isValid, errors] = validate(config, false);
  // if (!isValid) {
  //   throw new Error(errors.join('\n'));
  // }

  // return config;
  return {};
}

export function getFabMetadata(fabFile: PathLike | Buffer): [string, string] {
  /**
   * Extract the fab_id and fab_version from a FAB file or path.
   * @param fabFile The Flower App Bundle file to validate and extract the metadata from.
   *                It can either be a path to the file or the file itself as bytes.
   */
  const config = getFabConfig(fabFile);

  return [
    `${config['tool']['flwr']['app']['publisher']}/${config['project']['name']}`,
    config['project']['version'],
  ];
}

export function loadAndValidate(
  providedPath: PathLike | null = null,
  checkModule: boolean = true
): [AnyDict | null, string[], string[]] {
  /**
   * Load and validate pyproject.toml as a dictionary.
   * @param providedPath Optional path to pyproject.toml.
   * @param checkModule Whether to check module validity.
   */
  const configPath = providedPath ? path.resolve(providedPath.toString()) : path.join(process.cwd(), 'pyproject.toml');

  const config = load(configPath);

  if (!config) {
    return [
      null,
      ['Project configuration could not be loaded. `pyproject.toml` does not exist.'],
      [],
    ];
  }

  const [isValid, errors, warnings] = validate(config, checkModule, path.dirname(configPath));
  if (!isValid) {
    return [null, errors, warnings];
  }

  return [config, errors, warnings];
}

export function load(tomlPath: PathLike): AnyDict | null {
  /**
   * Load pyproject.toml and return as a dictionary.
   */
  if (!fs.existsSync(tomlPath)) {
    return null;
  }

  const tomlContent = fs.readFileSync(tomlPath, { encoding: 'utf-8' });
  return loadFromString(tomlContent);
}

function _validateRunConfig(configDict: AnyDict, errors: string[]): void {
  for (const [key, value] of Object.entries(configDict)) {
    if (typeof value === 'object' && !Array.isArray(value)) {
      _validateRunConfig(value, errors);
    }
    // else if (!getArgs(UserConfigValue).includes(typeof value)) {
    //   errors.push(
    //     `The value for key ${key} needs to be of type int, float, bool, string, or a dict of those.`
    //   );
    // }
  }
}

export function validateFields(config: AnyDict): [boolean, string[], string[]] {
  /**
   * Validate pyproject.toml fields.
   */
  const errors: string[] = [];
  const warnings: string[] = [];

  if (!config['project']) {
    errors.push('Missing [project] section');
  } else {
    if (!config['project']['name']) {
      errors.push('Property "name" missing in [project]');
    }
    if (!config['project']['version']) {
      errors.push('Property "version" missing in [project]');
    }
    if (!config['project']['description']) {
      warnings.push('Recommended property "description" missing in [project]');
    }
    if (!config['project']['license']) {
      warnings.push('Recommended property "license" missing in [project]');
    }
    if (!config['project']['authors']) {
      warnings.push('Recommended property "authors" missing in [project]');
    }
  }

  if (
    !config['tool'] ||
    !config['tool']['flwr'] ||
    !config['tool']['flwr']['app']
  ) {
    errors.push('Missing [tool.flwr.app] section');
  } else {
    if (!config['tool']['flwr']['app']['publisher']) {
      errors.push('Property "publisher" missing in [tool.flwr.app]');
    }
    if (config['tool']['flwr']['app']['config']) {
      _validateRunConfig(config['tool']['flwr']['app']['config'], errors);
    }
    if (!config['tool']['flwr']['app']['components']) {
      errors.push('Missing [tool.flwr.app.components] section');
    } else {
      if (!config['tool']['flwr']['app']['components']['serverapp']) {
        errors.push('Property "serverapp" missing in [tool.flwr.app.components]');
      }
      if (!config['tool']['flwr']['app']['components']['clientapp']) {
        errors.push('Property "clientapp" missing in [tool.flwr.app.components]');
      }
    }
  }

  return [errors.length === 0, errors, warnings];
}

export function validate(
  config: AnyDict,
  checkModule: boolean = true,
  projectDir: PathLike | null = null
): [boolean, string[], string[]] {
  /**
   * Validate pyproject.toml.
   */
  const [isValid, errors, warnings] = validateFields(config);

  if (!isValid) {
    return [false, errors, warnings];
  }

  // Validate serverapp
  const serverappRef = config['tool']['flwr']['app']['components']['serverapp'];
  // const [serverIsValid, serverReason] = objectRef.validate(serverappRef, checkModule, projectDir);

  // if (!serverIsValid && typeof serverReason === 'string') {
  //   return [false, [serverReason], []];
  // }

  // Validate clientapp
  const clientappRef = config['tool']['flwr']['app']['components']['clientapp'];
  // const [clientIsValid, clientReason] = objectRef.validate(clientappRef, checkModule, projectDir);

  // if (!clientIsValid && typeof clientReason === 'string') {
  //   return [false, [clientReason], []];
  // }

  return [true, [], []];
}

export function loadFromString(tomlContent: string): AnyDict | null {
  /**
   * Load TOML content from a string and return as a dictionary.
   */
  try {
    return toml.parse(tomlContent);
  } catch (error) {
    return null;
  }
}

// Get Flower home directory based on environment variables
export function getFlwrDir(providedPath?: string): string {
  if (!providedPath || !fs.existsSync(providedPath)) {
    return path.join(
      process.env[FLWR_HOME] || path.join(process.env['XDG_DATA_HOME'] || os.homedir(), '.flwr')
    );
  }
  return path.resolve(providedPath);
}

// Return the project directory based on fab_id and fab_version
export function getProjectDir(fabId: string, fabVersion: string, flwrDir?: string): string {
  if ((fabId.match(/\//g) || []).length !== 1) {
    throw new Error(`Invalid FAB ID: ${fabId}`);
  }

  const [publisher, projectName] = fabId.split('/');
  flwrDir = flwrDir || getFlwrDir();
  return path.join(flwrDir, APP_DIR, publisher, projectName, fabVersion);
}

// Return pyproject.toml configuration from the project directory
export function getProjectConfig(projectDir: string): { [key: string]: any } {
  const tomlPath = path.join(projectDir, FAB_CONFIG_FILE);
  if (!fs.existsSync(tomlPath)) {
    throw new Error(`Cannot find ${FAB_CONFIG_FILE} in ${projectDir}`);
  }

  const fileContents = fs.readFileSync(tomlPath, 'utf8');
  const config = toml.parse(fileContents);

  const [isValid, _warnings, errors] = validateFields(config);
  if (!isValid) {
    const errorMsg = errors.map((error: string) => `  - ${error}`).join('\n');
    throw new Error(`Invalid ${FAB_CONFIG_FILE}:\n${errorMsg}`);
  }

  return config;
}

// Merge a config with the overrides
export function fuseDicts(mainDict: UserConfig, overrideDict: UserConfig): UserConfig {
  const fusedDict = { ...mainDict };

  Object.entries(overrideDict).forEach(([key, value]) => {
    if (mainDict.hasOwnProperty(key)) {
      fusedDict[key] = value;
    }
  });

  return fusedDict;
}

// Merge overrides from a given dict with the config from a Flower App
export function getFusedConfigFromDir(projectDir: string, overrideConfig: UserConfig): UserConfig {
  const defaultConfig = getProjectConfig(projectDir)['tool']['flwr']['app']?.config || {};
  const flatDefaultConfig = flattenDict(defaultConfig);

  return fuseDicts(flatDefaultConfig, overrideConfig);
}

// Merge default config from a FAB with overrides in a Run
export function getFusedConfigFromFab(fabFile: string | Buffer, run: Run): UserConfig {
  const defaultConfig = getFabConfig(fabFile)['tool']['flwr']['app']?.config || {};
  const flatConfig = flattenDict(defaultConfig);

  return fuseDicts(flatConfig, run.overrideConfig);
}

// Merge overrides from a Run with the config from a FAB
export function getFusedConfig(run: Run, flwrDir?: string): UserConfig {
  if (!run.fabId || !run.fabVersion) {
    return {};
  }

  const projectDir = getProjectDir(run.fabId, run.fabVersion, flwrDir);

  if (!fs.existsSync(projectDir)) {
    return {};
  }

  return getFusedConfigFromDir(projectDir, run.overrideConfig);
}

// Flatten a nested dictionary by joining nested keys with a separator
export function flattenDict(rawDict: { [key: string]: any } | undefined, parentKey = ''): UserConfig {
  if (!rawDict) {
    return {};
  }

  const items: [string, UserConfigValue][] = [];
  const separator = '.';

  Object.entries(rawDict).forEach(([key, value]) => {
    const newKey = parentKey ? `${parentKey}${separator}${key}` : key;

    if (typeof value === 'object' && !Array.isArray(value)) {
      items.push(...Object.entries(flattenDict(value, newKey)));
    } else {
      items.push([newKey, value as UserConfigValue]);
    }
  });

  return Object.fromEntries(items);
}

// Unflatten a dictionary with keys containing separators into a nested dictionary
export function unflattenDict(flatDict: { [key: string]: any }): { [key: string]: any } {
  const unflattenedDict: { [key: string]: any } = {};
  const separator = '.';

  Object.entries(flatDict).forEach(([key, value]) => {
    const parts = key.split(separator);
    let current = unflattenedDict;

    parts.forEach((part, idx) => {
      if (idx === parts.length - 1) {
        current[part] = value;
      } else {
        if (!current[part]) {
          current[part] = {};
        }
        current = current[part];
      }
    });
  });

  return unflattenedDict;
}

// Parse a list of key-value pairs separated by '=' or load a TOML file
export function parseConfigArgs(config?: string[]): UserConfig {
  let overrides: UserConfig = {};

  if (!config) {
    return overrides;
  }

  if (config.length === 1 && config[0].endsWith('.toml')) {
    const fileContents = fs.readFileSync(config[0], 'utf8');
    overrides = flattenDict(toml.parse(fileContents));
    return overrides;
  }

  const pattern = /(\S+?)=(\'[^\']*\'|\"[^\"]*\"|\S+)/g;

  config.forEach((configLine) => {
    if (configLine) {
      if (configLine.endsWith('.toml')) {
        throw new Error('TOML files cannot be passed alongside key-value pairs.');
      }

      const matches = Array.from(configLine.matchAll(pattern));
      const tomlStr = matches.map(([_, k, v]) => `${k} = ${v}`).join('\n');
      Object.assign(overrides, toml.parse(tomlStr));
    }
  });

  return overrides;
}

// Extract `fab_version` and `fab_id` from a project config
export function getMetadataFromConfig(config: { [key: string]: any }): [string, string] {
  return [
    config['project']['version'],
    `${config['tool']['flwr']['app']['publisher']}/${config['project']['name']}`,
  ];
}
