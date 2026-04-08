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
import * as path from 'path';
import { tmpdir } from 'os';
import {
  flattenDict,
  fuseDicts,
  getFlwrDir,
  getProjectConfig,
  getProjectDir,
  parseConfigArgs,
  unflattenDict,
} from './config';
import { UserConfig } from './typing';

// Mock constants
const FAB_CONFIG_FILE = 'pyproject.toml';

describe('Configuration Utilities', () => {

  beforeEach(() => {
    jest.resetModules();
  });

  it('test_get_flwr_dir_with_provided_path', () => {
    const providedPath = '.';
    expect(getFlwrDir(providedPath)).toBe(path.resolve(providedPath));
  });

  it('test_get_flwr_dir_without_provided_path', () => {
    jest.spyOn(process, 'env', 'get').mockReturnValue({ HOME: '/home/user' });
    expect(getFlwrDir()).toBe(path.join('/home/user', '.flwr'));
  });

  it('test_get_flwr_dir_with_flwr_home', () => {
    jest.spyOn(process, 'env', 'get').mockReturnValue({ FLWR_HOME: '/custom/flwr/home' });
    expect(getFlwrDir()).toBe(path.join('/custom/flwr/home'));
  });

  it('test_get_flwr_dir_with_xdg_data_home', () => {
    jest.spyOn(process, 'env', 'get').mockReturnValue({ XDG_DATA_HOME: '/custom/data/home' });
    expect(getFlwrDir()).toBe(path.join('/custom/data/home', '.flwr'));
  });

  it('test_get_project_dir_invalid_fab_id', () => {
    expect(() => {
      getProjectDir('invalid_fab_id', '1.0.0');
    }).toThrow(Error);
  });

  it('test_get_project_dir_valid', () => {
    const appPath = getProjectDir('app_name/user', '1.0.0', '.');
    expect(appPath).toBe(path.join('.', 'apps', 'app_name', 'user', '1.0.0'));
  });

  // it('test_get_project_config_file_not_found', () => {
  //   expect(() => {
  //     getProjectConfig('/invalid/dir');
  //   }).toThrow(Error);
  // });

  // it('test_get_fused_config_valid', () => {
  //   const pyprojectTomlContent = `
  //     [build-system]
  //     requires = ["hatchling"]
  //     build-backend = "hatchling.build"

  //     [project]
  //     name = "fedgpt"
  //     version = "1.0.0"

  //     [tool.flwr.app]
  //     publisher = "flwrlabs"

  //     [tool.flwr.app.config]
  //     num_server_rounds = 10
  //     momentum = 0.1
  //     lr = 0.01
  //     progress_bar = true
  //   `;
  //   const overrides: UserConfig = {
  //     num_server_rounds: 5,
  //     lr: 0.2,
  //     "serverapp.test": "overriden",
  //   };
  //   const expectedConfig = {
  //     num_server_rounds: 5,
  //     momentum: 0.1,
  //     lr: 0.2,
  //     progress_bar: true,
  //     "serverapp.test": "overriden",
  //     "clientapp.test": "key",
  //   };

  //   const tmpPath = path.join(tmpdir(), 'project_dir');
  //   fs.mkdirSync(tmpPath);
  //   const tomlPath = path.join(tmpPath, FAB_CONFIG_FILE);
  //   fs.writeFileSync(tomlPath, pyprojectTomlContent);

  //   try {
  //     const defaultConfig = getProjectConfig(tmpPath)['tool']['flwr']['app'].config || {};
  //     const config = fuseDicts(flattenDict(defaultConfig), overrides);

  //     expect(config).toEqual(expectedConfig);
  //   } finally {
  //     fs.rmdirSync(tmpPath, { recursive: true });
  //   }
  // });

  // it('test_flatten_dict', () => {
  //   const rawDict = { a: { b: { c: 'd' } }, e: 'f' };
  //   const expected = { 'a.b.c': 'd', e: 'f' };
  //   expect(flattenDict(rawDict)).toEqual(expected);
  // });

  // it('test_unflatten_dict', () => {
  //   const rawDict = { 'a.b.c': 'd', e: 'f' };
  //   const expected = { a: { b: { c: 'd' } }, e: 'f' };
  //   expect(unflattenDict(rawDict)).toEqual(expected);
  // });

  // it('test_parse_config_args_none', () => {
  //   expect(parseConfigArgs(undefined)).toEqual({});
  // });

  // it('test_parse_config_args_overrides', () => {
  //   const config = parseConfigArgs([
  //     "key1='value1' key2='value2'",
  //     'key3=1',
  //     "key4=2.0 key5=true key6='value6'",
  //   ]);
  //   const expected = {
  //     key1: 'value1',
  //     key2: 'value2',
  //     key3: 1,
  //     key4: 2.0,
  //     key5: true,
  //     key6: 'value6',
  //   };
  //   expect(config).toEqual(expected);
  // });

  // it('test_parse_config_args_from_toml_file', () => {
  //   const tomlConfig = `
  //     num_server_rounds = 10
  //     momentum = 0.1
  //     verbose = true
  //   `;

  //   const initialRunConfig: UserConfig = {
  //     "num_server_rounds": 5,
  //     "momentum": 0.2,
  //     "dataset": "my-fancy-dataset",
  //     "verbose": false,
  //   };
  //   const expectedConfig = {
  //     "num_server_rounds": 10,
  //     "momentum": 0.1,
  //     "dataset": "my-fancy-dataset",
  //     "verbose": true,
  //   };

  //   const tmpPath = tmpdir();
  //   const tomlConfigFile = path.join(tmpPath, 'extra_config.toml');
  //   fs.writeFileSync(tomlConfigFile, tomlConfig);

  //   const configFromToml = parseConfigArgs([tomlConfigFile]);
  //   const config = fuseDicts(initialRunConfig, configFromToml);

  //   expect(config).toEqual(expectedConfig);

  //   fs.unlinkSync(tomlConfigFile);
  // });

  // it('test_parse_config_args_passing_toml_and_key_value', () => {
  //   const config = ['my-other-config.toml', 'lr=0.1', 'epochs=99'];
  //   expect(() => {
  //     parseConfigArgs(config);
  //   }).toThrow(Error);
  // });
});
