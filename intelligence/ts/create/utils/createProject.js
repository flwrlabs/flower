import fs from 'fs/promises';
import path from 'node:path';
import chalk from 'chalk';
import { execSync } from 'node:child_process';

const updateFile = async (filename, projectPath, requestedPackage, projectName) => {
  console.log(chalk.gray(`Updating ${filename}...`));

  const packagePath = path.join(projectPath, filename);
  const stat = await fs.lstat(packagePath);

  if (stat.isFile()) {
    const currentPackageJson = await fs.readFile(packagePath, 'utf8');
    const newFileContent = currentPackageJson.replace(requestedPackage, projectName);
    await fs.writeFile(packagePath, newFileContent, 'utf8');
  }
};

export const init = async (projectName, requestedPackage) => {
  try {
    const currentDir = process.cwd();
    const destination = path.join(currentDir, projectName);
    const tempDir = path.join(currentDir, `temp-${projectName}`);
    const fullURL = `https://github.com/flwrlabs/flower.git`;

    console.log(chalk.gray('Cloning template repository into a temporary folder...'));
    execSync(`git clone --depth 1 ${fullURL} ${tempDir}`, {
      stdio: 'inherit',
    });

    // Define the path to the template inside the cloned repo
    const templatePath = path.join(tempDir, 'intelligence', 'ts', 'templates', requestedPackage);

    console.log(chalk.gray('Copying template files to project destination...'));
    // Create destination folder and copy the template files into it
    await fs.mkdir(destination, { recursive: true });
    await fs.cp(templatePath, destination, { recursive: true });

    // Remove the temporary cloned repository
    await fs.rm(tempDir, { recursive: true, force: true });

    console.log(chalk.gray('📑  Files copied...'));

    await updateFile('package.json', destination, requestedPackage, projectName);
    await updateFile('README.md', destination, requestedPackage, projectName);

    console.log(chalk.green(`\ncd ${projectName}\npnpm start`));
  } catch (error) {
    console.log(chalk.red(error));
  }
};
