/**
 * Python Environment Checker
 * Ensures Python and dependencies are installed before starting backend
 */

const { execSync, spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

class PythonChecker {
  constructor() {
    this.pythonPath = null;
    this.pipPath = null;
  }

  /**
   * Check if Python 3.9+ is installed
   */
  checkPython() {
    console.log('Checking for Python installation...');
    
    const pythonCommands = ['python', 'python3', 'py'];
    
    for (const cmd of pythonCommands) {
      try {
        const version = execSync(`${cmd} --version`, { encoding: 'utf-8', stdio: ['pipe', 'pipe', 'ignore'] });
        console.log(`Found: ${version.trim()}`);
        
        // Check version is 3.9+
        const match = version.match(/Python (\d+)\.(\d+)/);
        if (match) {
          const major = parseInt(match[1]);
          const minor = parseInt(match[2]);
          
          if (major === 3 && minor >= 9) {
            this.pythonPath = cmd;
            this.pipPath = `${cmd} -m pip`;
            console.log(`✓ Python ${major}.${minor} found: ${cmd}`);
            return true;
          } else if (major > 3) {
            this.pythonPath = cmd;
            this.pipPath = `${cmd} -m pip`;
            console.log(`✓ Python ${major}.${minor} found: ${cmd}`);
            return true;
          }
        }
      } catch (error) {
        // Command not found, try next
        continue;
      }
    }
    
    console.log('✗ Python 3.9+ not found');
    return false;
  }

  /**
   * Check if backend dependencies are installed
   */
  checkDependencies() {
    if (!this.pythonPath) {
      return false;
    }

    console.log('Checking Python dependencies...');
    
    try {
      // Check if key packages are installed
      const packages = ['torch', 'fastapi', 'opencv-python', 'numpy'];
      
      for (const pkg of packages) {
        try {
          execSync(`${this.pythonPath} -c "import ${pkg.replace('-', '_')}"`, { 
            stdio: 'ignore' 
          });
        } catch (error) {
          console.log(`✗ Missing package: ${pkg}`);
          return false;
        }
      }
      
      console.log('✓ All dependencies installed');
      return true;
    } catch (error) {
      console.log('✗ Dependency check failed:', error.message);
      return false;
    }
  }

  /**
   * Install backend dependencies
   */
  async installDependencies(progressCallback) {
    if (!this.pythonPath) {
      throw new Error('Python not found');
    }

    const backendPath = path.join(__dirname, '..', 'local-ai-backend');
    const requirementsPath = path.join(backendPath, 'requirements.txt');

    if (!fs.existsSync(requirementsPath)) {
      throw new Error('requirements.txt not found');
    }

    console.log('Installing Python dependencies...');
    console.log('This may take 5-10 minutes on first run...');

    return new Promise((resolve, reject) => {
      const pip = spawn(this.pythonPath, [
        '-m', 'pip', 'install', '-r', requirementsPath
      ], {
        cwd: backendPath,
        stdio: ['ignore', 'pipe', 'pipe']
      });

      let output = '';

      pip.stdout.on('data', (data) => {
        const text = data.toString();
        output += text;
        console.log(text.trim());
        
        if (progressCallback) {
          // Parse pip output for progress
          if (text.includes('Collecting') || text.includes('Downloading')) {
            const match = text.match(/(\S+)/);
            if (match) {
              progressCallback({ status: 'downloading', package: match[1] });
            }
          } else if (text.includes('Installing')) {
            const match = text.match(/Installing collected packages: (.+)/);
            if (match) {
              progressCallback({ status: 'installing', packages: match[1] });
            }
          }
        }
      });

      pip.stderr.on('data', (data) => {
        const text = data.toString();
        output += text;
        console.error(text.trim());
      });

      pip.on('close', (code) => {
        if (code === 0) {
          console.log('✓ Dependencies installed successfully');
          resolve(true);
        } else {
          console.error('✗ Dependency installation failed');
          reject(new Error(`pip install exited with code ${code}\n${output}`));
        }
      });

      pip.on('error', (error) => {
        reject(error);
      });
    });
  }

  /**
   * Get Python download URL
   */
  getPythonDownloadUrl() {
    return 'https://www.python.org/downloads/';
  }
}

module.exports = new PythonChecker();
