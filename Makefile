clean:
	rm -rf build
	rm -rf dist
	rm -rf fairness.egg-info

testdist:
	rm -rf dist
	rm -rf build
	python3 setup.py sdist bdist_wheel
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*

productiondist:
	rm -rf dist
	rm -rf build
	python3 setup.py sdist bdist_wheel
	twine upload dist/*

