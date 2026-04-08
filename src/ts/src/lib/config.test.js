"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || function (mod) {
    if (mod && mod.__esModule) return mod;
    var result = {};
    if (mod != null) for (var k in mod) if (k !== "default" && Object.prototype.hasOwnProperty.call(mod, k)) __createBinding(result, mod, k);
    __setModuleDefault(result, mod);
    return result;
};
Object.defineProperty(exports, "__esModule", { value: true });
const path = __importStar(require("path"));
const config_1 = require("./config");
// Mock constants
const FAB_CONFIG_FILE = 'pyproject.toml';
describe('Configuration Utilities', () => {
    beforeEach(() => {
        jest.resetModules();
    });
    it('test_get_flwr_dir_with_provided_path', () => {
        const providedPath = '.';
        expect((0, config_1.getFlwrDir)(providedPath)).toBe(path.resolve(providedPath));
    });
    it('test_get_flwr_dir_without_provided_path', () => {
        jest.spyOn(process, 'env', 'get').mockReturnValue({ HOME: '/home/user' });
        expect((0, config_1.getFlwrDir)()).toBe(path.join('/home/user', '.flwr'));
    });
    it('test_get_flwr_dir_with_flwr_home', () => {
        jest.spyOn(process, 'env', 'get').mockReturnValue({ FLWR_HOME: '/custom/flwr/home' });
        expect((0, config_1.getFlwrDir)()).toBe(path.join('/custom/flwr/home'));
    });
    it('test_get_flwr_dir_with_xdg_data_home', () => {
        jest.spyOn(process, 'env', 'get').mockReturnValue({ XDG_DATA_HOME: '/custom/data/home' });
        expect((0, config_1.getFlwrDir)()).toBe(path.join('/custom/data/home', '.flwr'));
    });
    it('test_get_project_dir_invalid_fab_id', () => {
        expect(() => {
            (0, config_1.getProjectDir)('invalid_fab_id', '1.0.0');
        }).toThrow(Error);
    });
    it('test_get_project_dir_valid', () => {
        const appPath = (0, config_1.getProjectDir)('app_name/user', '1.0.0', '.');
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
