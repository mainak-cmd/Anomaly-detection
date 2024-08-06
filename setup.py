from setuptools import setup, find_packages

setup(
    name='anamolyDetection',
    version='0.1.0',  # Semantic versioning
    packages=find_packages(),  # Automatically find packages in the directory
    install_requires=[
        'numpy',  # List your dependencies here
        # Add any other dependencies, e.g., 'scikit-learn', 'pandas', etc.
    ],
    include_package_data=True,  # Include non-Python files specified in MANIFEST.in
    description='A module for anomaly detection.',
    author='Mainak',
    author_email='mainakr748@gmail.com',
    url='https://github.com/yourusername/anamolyDetection',  # Replace with your URL
    license='MIT',  # Choose an appropriate license
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  # Specify the Python versions you support
)
